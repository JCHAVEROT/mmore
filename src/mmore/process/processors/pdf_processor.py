import io
import logging
import math
import os
import queue
import re
from dataclasses import dataclass, field
from multiprocessing import Manager, Process, set_start_method
from typing import Any, Dict, List, Optional, Tuple, cast

import pymupdf
import torch
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from PIL import Image, UnidentifiedImageError

from ...type import DocumentMetadata, FileDescriptor, MultimodalSample
from ..utils import clean_image, clean_text
from .base import Processor, ProcessorConfig

IMG_REGEX = r"!\[\]\(_page_\d+_[A-Za-z0-9_]+\.(jpeg|jpg|png|gif)\)"


@dataclass
class PDFMetadata(DocumentMetadata):
    paragraph_starts: List[Tuple[int, int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        metadata = super().to_dict()
        if self.paragraph_starts:
            metadata["paragraph_starts"] = self.paragraph_starts

        return metadata


class PDFProcessor(Processor):
    artifact_dict = None
    _cache_warmed = False

    def __init__(self, config=None):
        super().__init__(config=config or ProcessorConfig())
        self.converter = None

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() == ".pdf"

    @staticmethod
    def _pdftext_workers() -> int:
        """Number of CPU workers marker uses for PDF text extraction.

        Marker's auto-detection underutilizes high-core nodes, so we set it
        explicitly. Defaults to half the cores to leave headroom for the outer
        per-file pool; override with PDFTEXT_WORKERS.
        """
        return int(
            os.environ.get("PDFTEXT_WORKERS", max(1, (os.cpu_count() or 2) // 2))
        )

    @staticmethod
    def _prewarm_model_cache() -> None:
        """Populate the on-disk model cache once in the parent process.

        Without this, every spawned GPU worker calls create_model_dict() on a
        cold cache simultaneously and they race to download the same files
        (surya then errors with "destination already exists" and retries).
        Loading on CPU only touches the disk cache without occupying GPU memory.
        """
        if PDFProcessor._cache_warmed:
            return
        create_model_dict(device="cpu")
        PDFProcessor._cache_warmed = True

    @staticmethod
    def _gather_parallel_results(
        processes, output_queue, error_queue, join_timeout=120
    ):
        """Collect exactly one payload per worker, then shut the workers down.

        marker spawns its own internal worker pool, so a child process may not
        exit promptly after returning its result. Spin-waiting on is_alive()
        therefore hangs forever; instead we collect all expected payloads, then
        join with a timeout and terminate any straggler.
        """
        payloads = []
        for _ in processes:
            while True:
                if not error_queue.empty():
                    raise RuntimeError(f"Child process failed: {error_queue.get()}")
                try:
                    payloads.append(output_queue.get(timeout=1.0))
                    break
                except queue.Empty:
                    if (
                        not any(p.is_alive() for p in processes)
                        and output_queue.empty()
                    ):
                        if not error_queue.empty():
                            raise RuntimeError(
                                f"Child process failed: {error_queue.get()}"
                            )
                        raise RuntimeError(
                            "A PDF worker exited without returning a result"
                        )

        for p in processes:
            p.join(timeout=join_timeout)
            if p.is_alive():
                logging.warning(
                    "PDF worker still alive after returning its result; terminating."
                )
                p.terminate()
                p.join(timeout=10)
        return payloads

    @staticmethod
    def load_models(disable_image_extraction: bool = False):
        if PDFProcessor.artifact_dict is None:
            PDFProcessor.artifact_dict = create_model_dict()

        marker_config = {
            "disable_image_extraction": disable_image_extraction,
            "languages": None,
            "use_llm": False,
            "disable_multiprocessing": False,
            "paginate_output": True,
            "pdftext_workers": PDFProcessor._pdftext_workers(),
        }
        config_parser = ConfigParser(marker_config)
        converter = PdfConverter(
            artifact_dict=PDFProcessor.artifact_dict,
            config=config_parser.generate_config_dict(),
        )

        converter.initialize_processors(list(converter.default_processors))

        return converter

    # overwriting the process_batch
    def process_batch(
        self, files_paths: List[str], fast_mode: bool = False, num_workers: int = 1
    ) -> List[MultimodalSample]:
        if fast_mode:  # No GPU available - fallback to default
            return super().process_batch(files_paths, fast_mode, num_workers)
        else:
            if not torch.cuda.is_available():
                num_gpus = 1
            else:
                num_gpus = torch.cuda.device_count()

            # Single GPU (or CPU): parallelize across files with the shared pool.
            if num_gpus <= 1:
                # Multiple files: let the shared worker pool process them in
                # parallel. The converter stays None on the parent so the
                # processor pickles cheaply; each worker lazily builds its own.
                if self._pool is not None and len(files_paths) > 1:
                    return self._pool.map(self.process, files_paths)

                if self.converter is None:
                    self.converter = PDFProcessor.load_models(
                        disable_image_extraction=not self.config.custom_config.get(
                            "extract_images", True
                        )
                    )

                results = []
                for file_path in files_paths:
                    try:
                        res = self.process(file_path)
                        results.append(res)
                    except Exception as e:
                        logging.error(f"Failed to process {file_path}: {str(e)}")

                return results
            else:  # Multiple GPUs available
                # A single large PDF can't be split file-wise, so split it
                # page-wise across GPUs instead of leaving all but one idle.
                if len(files_paths) == 1:
                    page_threshold = int(
                        self.config.custom_config.get("multi_gpu_page_threshold", 50)
                    )
                    if self._pdf_page_count(files_paths[0]) > page_threshold:
                        return [
                            self._process_single_file_multi_gpu(
                                files_paths[0], num_gpus, self.config.custom_config
                            )
                        ]

                batches = self._split_files(files_paths, num_gpus)

                try:
                    set_start_method("spawn", force=True)
                except RuntimeError:
                    pass

                self._prewarm_model_cache()

                manager = Manager()
                output_queue = manager.Queue()
                error_queue = manager.Queue()
                processes = []

                for i, batch in enumerate(batches):
                    if not batch:
                        continue
                    gpu_id = i % num_gpus
                    p = Process(
                        target=self._process_parallel,
                        args=(
                            batch,
                            gpu_id,
                            self.config.custom_config,
                            output_queue,
                            error_queue,
                        ),
                    )
                    processes.append(p)
                    p.start()

                results = []
                for batch_results in self._gather_parallel_results(
                    processes, output_queue, error_queue
                ):
                    results.extend(batch_results)

                return results

    # Regex matching marker page separators: \n\n{page_id}----...\n\n
    _PAGE_SEP_RE = re.compile(r"\n\n\{(\d+)\}-{3,}\n\n")

    def process(self, file_path: str) -> MultimodalSample:
        if self.converter is None:
            self.converter = PDFProcessor.load_models(
                disable_image_extraction=not self.config.custom_config.get(
                    "extract_images", True
                )
            )

        rendered = self.converter(file_path)
        text, _, images = text_from_rendered(rendered)
        text = re.sub(str(IMG_REGEX), "<attachment>", cast(str, text))
        images = list(images.values())

        paragraph_starts, text = self._parse_pagination(cast(str, text))

        metadata = PDFMetadata(file_path=file_path)
        if paragraph_starts:
            metadata.paragraph_starts = paragraph_starts

        return self.create_sample([text], images, metadata)

    @classmethod
    def _parse_pagination(
        cls, text: str
    ) -> Tuple[
        List[Tuple[int, int, int]],
        str,
    ]:
        """Parse marker pagination separators to build paragraph_starts,
        then strip the separators from the text."""
        separators = list(cls._PAGE_SEP_RE.finditer(text))
        if not separators:
            return [], text

        page_texts: List[Tuple[int, str]] = []  # (page_id, page_content)
        prev_end = 0
        for match in separators:
            page_id = int(match.group(1))
            page_content = text[prev_end : match.start()]
            page_texts.append((page_id, page_content))
            prev_end = match.end()
        trailing = text[prev_end:]
        if trailing.strip():
            last_page_id = int(separators[-1].group(1)) + 1
            page_texts.append((last_page_id, trailing))

        paragraph_starts: List[Tuple[int, int, int]] = []
        current_position = 0

        for page_id, page_content in page_texts:
            para_idx = 0
            offset_in_page = 0
            for segment in page_content.split("\n\n"):
                if segment.strip():
                    paragraph_starts.append(
                        (current_position + offset_in_page, page_id, para_idx)
                    )
                    para_idx += 1
                offset_in_page += len(segment) + 2

            current_position += len(page_content)

        paragraph_starts.append((current_position, -1, -1))

        clean_text = "".join(content for _, content in page_texts)

        return paragraph_starts, clean_text

    def process_fast(self, file_path: str) -> MultimodalSample:
        pdf_doc = pymupdf.Document(file_path)
        all_text_parts = []
        embedded_images = []
        paragraph_starts: List[
            Tuple[int, int, int]
        ] = []  # (char_offset, page_num, para_index)
        current_position = 0

        def _extract_images(pdf_doc, xref) -> Optional[Image.Image]:
            try:
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image.get("image")

                if image_bytes is None:
                    logging.error(f"No image data found for xref {xref}")

                return Image.open(io.BytesIO(cast(bytes, image_bytes))).convert("RGB")

            except KeyError as e:
                logging.error(f"KeyError while extracting image: {e}")
                return None

            except UnidentifiedImageError as e:
                logging.error(
                    f"UnidentifiedImageError: Could not identify image file for xref {xref}: {e}"
                )
                return None

            except Exception as e:
                logging.error(
                    f"Unexpected error while extracting image for xref {xref}: {e}"
                )
                return None

        for page_num, page in enumerate(pdf_doc):  # pyright: ignore[reportArgumentType]
            text = clean_text(page.get_text())  # type: ignore[attr-defined]

            if text.strip():
                para_idx = 0
                offset_in_page = 0
                for segment in text.split("\n\n"):
                    if segment.strip():
                        paragraph_starts.append(
                            (current_position + offset_in_page, page_num, para_idx)
                        )
                        para_idx += 1
                    offset_in_page += len(segment) + 2  # +2 for the "\n\n" separator

                all_text_parts.append(text)
                current_position += len(text)

            if self.config.custom_config.get("extract_images", True):
                for img_info in page.get_images(full=False):
                    image = _extract_images(pdf_doc, img_info[0])
                    if image and clean_image(image):
                        # clean image filters images below size 512x512 and variance below 100, these are defaults and can be changed
                        embedded_images.append(image)
                        attachment_text = self.config.attachment_tag
                        all_text_parts.append(attachment_text)
                        current_position += len(attachment_text)
            else:
                embedded_images = []

        paragraph_starts.append((current_position, -1, -1))
        metadata = PDFMetadata(file_path=file_path, paragraph_starts=paragraph_starts)

        full_text = "".join(all_text_parts)
        return self.create_sample([full_text], embedded_images, metadata)

    # Functions for parallelizing across GPUs
    def _split_files(self, files_paths, num_batches):
        file_sizes = [(file, self.get_file_size(file)) for file in files_paths]
        sorted_files = sorted(file_sizes, key=lambda x: x[1], reverse=True)

        batches = [[] for _ in range(num_batches)]
        batch_sizes = [0] * num_batches

        for file, size in sorted_files:
            min_index = batch_sizes.index(min(batch_sizes))
            batches[min_index].append(file)
            batch_sizes[min_index] += size

        batches = [batch for batch in batches if batch]
        return batches

    def _process_parallel(
        self, files_paths, gpu_id, config_custom, output_queue, error_queue
    ):
        try:
            torch.cuda.set_device(gpu_id)

            if PDFProcessor.artifact_dict is None:
                PDFProcessor.artifact_dict = create_model_dict()

            marker_config = {
                "disable_image_extraction": not config_custom.get(
                    "extract_images", True
                ),
                "languages": None,
                "use_llm": False,
                "disable_multiprocessing": False,
                "paginate_output": True,
                "device": f"cuda:{gpu_id}",
                "pdftext_workers": PDFProcessor._pdftext_workers(),
            }

            config_parser = ConfigParser(marker_config)
            self.converter = PdfConverter(
                artifact_dict=PDFProcessor.artifact_dict,
                config=config_parser.generate_config_dict(),
            )

            batch_results = []
            for file in files_paths:
                try:
                    result = self.process(file)
                    batch_results.append(result)
                except Exception as e:
                    logging.error(f"Failed to process {file}: {str(e)}")
                    batch_results.append(None)  # handle partial failures

            output_queue.put(batch_results)

        except Exception as e:
            error_queue.put(f"GPU {gpu_id} failed: {str(e)}")
            raise e
        finally:
            torch.cuda.empty_cache()
            if hasattr(self, "converter"):
                del self.converter

    # Functions for parallelizing a single large PDF across GPUs
    @staticmethod
    def _pdf_page_count(file_path: str) -> int:
        doc = pymupdf.Document(file_path)
        try:
            return doc.page_count
        finally:
            doc.close()

    def _process_single_file_multi_gpu(
        self, file_path: str, num_gpus: int, config_custom: Dict[str, Any]
    ) -> MultimodalSample:
        """Process a single large PDF by splitting its pages into contiguous
        ranges, one per GPU, then merging the results in page order."""
        total_pages = self._pdf_page_count(file_path)
        pages_per_gpu = math.ceil(total_pages / num_gpus)

        ranges: List[Tuple[int, int]] = []
        for i in range(num_gpus):
            start = i * pages_per_gpu
            end = min(start + pages_per_gpu, total_pages)
            if start < end:
                ranges.append((start, end))

        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        self._prewarm_model_cache()

        manager = Manager()
        output_queue = manager.Queue()
        error_queue = manager.Queue()
        processes = []

        for batch_num, (start, end) in enumerate(ranges):
            gpu_id = batch_num % num_gpus
            p = Process(
                target=self._process_page_range,
                args=(
                    file_path,
                    start,
                    end,
                    gpu_id,
                    config_custom,
                    output_queue,
                    error_queue,
                    batch_num,
                ),
            )
            processes.append(p)
            p.start()

        results: Dict[int, Tuple[int, str, List[Image.Image]]] = {}
        for batch_num, payload in self._gather_parallel_results(
            processes, output_queue, error_queue
        ):
            results[batch_num] = payload

        ordered = [results[i] for i in range(len(ranges)) if i in results]
        return self._merge_page_range_results(ordered, file_path)

    @staticmethod
    def _process_page_range(
        file_path: str,
        start_page: int,
        end_page: int,
        gpu_id: int,
        config_custom: Dict[str, Any],
        output_queue,
        error_queue,
        batch_num: int = 0,
    ) -> None:
        """Process pages [start_page, end_page) of a PDF on a specific GPU.

        Returns the raw (paginated) marker text plus images via the queue; the
        parent merges and assigns absolute page numbers. Parsing is deferred to
        the parent so paragraph offsets are computed against the merged text.
        """
        try:
            torch.cuda.set_device(gpu_id)

            if PDFProcessor.artifact_dict is None:
                PDFProcessor.artifact_dict = create_model_dict()

            marker_config = {
                "disable_image_extraction": not config_custom.get(
                    "extract_images", True
                ),
                "languages": None,
                "use_llm": False,
                "disable_multiprocessing": False,
                "paginate_output": True,
                "device": f"cuda:{gpu_id}",
                "page_range": list(range(start_page, end_page)),
                "pdftext_workers": PDFProcessor._pdftext_workers(),
            }

            config_parser = ConfigParser(marker_config)
            converter = PdfConverter(
                artifact_dict=PDFProcessor.artifact_dict,
                config=config_parser.generate_config_dict(),
            )

            rendered = converter(file_path)
            text, _, images = text_from_rendered(rendered)
            text = re.sub(str(IMG_REGEX), "<attachment>", cast(str, text))

            output_queue.put((batch_num, (start_page, text, list(images.values()))))

        except Exception as e:
            error_queue.put(
                f"GPU {gpu_id} page range {start_page}-{end_page}: {str(e)}"
            )
            raise e
        finally:
            torch.cuda.empty_cache()

    def _merge_page_range_results(
        self,
        ordered: List[Tuple[int, str, List[Image.Image]]],
        file_path: str,
    ) -> MultimodalSample:
        """Merge page-range results into one sample, in page order.

        Concatenates the cleaned page texts and rebuilds ``paragraph_starts``
        with cumulative character offsets and absolute page numbers. Marker's
        per-range page ids (whether range-relative or absolute) are remapped to
        contiguous absolute pages anchored at each range's start page.
        """
        all_clean: List[str] = []
        all_images: List[Image.Image] = []
        merged_starts: List[Tuple[int, int, int]] = []
        char_offset = 0

        for start_page, raw_text, images in ordered:
            starts, clean = self._parse_pagination(raw_text)
            all_images.extend(images)

            page_remap: Dict[int, int] = {}
            next_page = start_page
            for offset, page_id, para_idx in starts:
                if page_id == -1:  # per-range terminal marker, re-added globally
                    continue
                if page_id not in page_remap:
                    page_remap[page_id] = next_page
                    next_page += 1
                merged_starts.append(
                    (char_offset + offset, page_remap[page_id], para_idx)
                )

            char_offset += len(clean)
            all_clean.append(clean)

        merged_starts.append((char_offset, -1, -1))

        metadata = PDFMetadata(file_path=file_path, paragraph_starts=merged_starts)
        return self.create_sample(["".join(all_clean)], all_images, metadata)
