"""
File Validation Service - Handles all file validation logic for claim processing.
This service ensures uploaded files meet requirements before processing.
"""

from typing import List

from fastapi import HTTPException, status

from app.config.settings import Config
from app.core.logger import logger


class FileValidator:
    """
    Service for validating uploaded files before processing.

    Validates:
    - File count limits
    - File types
    - File sizes
    - Filename requirements
    """

    def __init__(self):
        self.max_files = Config.MAX_FILES_PER_REQUEST
        self.max_file_size_mb = Config.MAX_FILE_SIZE_MB
        self.supported_types = Config.SUPPORTED_FILE_TYPES

    async def validate_files(self, files: List[bytes], filenames: List[str]) -> None:
        """
        Validate a list of files and filenames.

        Args:
            files: List of file contents as bytes
            filenames: List of corresponding filenames

        Raises:
            HTTPException: If validation fails
        """
        # Validate file count
        self._validate_file_count(len(files))

        # Validate each file
        for file_content, filename in zip(files, filenames):
            await self.validate_file(file_content, filename)

    async def validate_file(self, file_content: bytes, filename: str) -> None:
        """
        Validate a single file.

        Args:
            file_content: File content as bytes
            filename: Name of the file

        Raises:
            HTTPException: If validation fails
        """
        # Validate filename
        self._validate_filename(filename)

        # Validate file size
        self._validate_file_size(file_content, filename)

        # Validate file type (basic check - content type would be better)
        self._validate_file_type(file_content, filename)

        logger.info(f"File validation passed: {filename}")

    def _validate_file_count(self, file_count: int) -> None:
        """Validate the number of files."""
        if file_count > self.max_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many files. Maximum {self.max_files} files allowed per request.",
            )

        if file_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one file must be uploaded.",
            )

    def _validate_filename(self, filename: str) -> None:
        """Validate filename requirements."""
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is missing a filename.",
            )

        if len(filename) > 255:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Filename too long: {filename}",
            )

        # Check for potentially dangerous characters
        dangerous_chars = ["<", ">", ":", '"', "|", "?", "*", "\\", "/"]
        if any(char in filename for char in dangerous_chars):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Filename contains invalid characters: {filename}",
            )

    def _validate_file_size(self, file_content: bytes, filename: str) -> None:
        """Validate file size."""
        file_size_mb = len(file_content) / (1024 * 1024)

        if file_size_mb > self.max_file_size_mb:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {filename} is too large. Maximum {self.max_file_size_mb}MB allowed.",
            )

        if file_size_mb == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {filename} is empty.",
            )

    def _validate_file_type(self, file_content: bytes, filename: str) -> None:
        """Basic file type validation based on content."""
        # Check if it's a PDF by looking for PDF header
        if not file_content.startswith(b"%PDF"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {filename} is not a valid PDF file.",
            )

    def get_file_info(self, file_content: bytes, filename: str) -> dict:
        """Get information about a file for logging/debugging."""
        return {
            "filename": filename,
            "size_bytes": len(file_content),
            "size_mb": len(file_content) / (1024 * 1024),
            "is_pdf": file_content.startswith(b"%PDF"),
        }
