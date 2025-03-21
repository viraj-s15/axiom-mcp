"""Tests for advanced resource types and utilities."""

import json

import pytest
from pydantic import AnyUrl

from axiom_mcp.resources.advanced_types import (
    CompressedResource,
    CompressionConfig,
    CompressionError,
    TransformationPipeline,
    TransformedResource,
    bytes_transform,
    compress_content,
    create_json_pipeline,
    decompress_content,
    safe_decode,
    safe_encode,
    text_transform,
)
from axiom_mcp.resources.types import TextResource


@pytest.fixture
def sample_text():
    return "Hello, World!" * 100  # Make it long enough to be worth compressing


@pytest.fixture
def text_resource(sample_text):
    return TextResource(
        uri=AnyUrl("memory://test/text"), text=sample_text, name="test_text"
    )


class TestCompressionUtils:
    def test_compress_decompress_gzip(self, sample_text):
        """Test gzip compression and decompression."""
        compressed = compress_content(sample_text, "gzip")
        assert isinstance(compressed, bytes)
        assert len(compressed) < len(sample_text)
        decompressed = decompress_content(compressed, "gzip")
        assert decompressed.decode() == sample_text

    def test_compress_decompress_lzma(self, sample_text):
        """Test LZMA compression and decompression."""
        compressed = compress_content(sample_text, "lzma")
        assert isinstance(compressed, bytes)
        assert len(compressed) < len(sample_text)
        decompressed = decompress_content(compressed, "lzma")
        assert decompressed.decode() == sample_text

    def test_compress_decompress_zlib(self, sample_text):
        """Test zlib compression and decompression."""
        compressed = compress_content(sample_text, "zlib")
        assert isinstance(compressed, bytes)
        assert len(compressed) < len(sample_text)
        decompressed = decompress_content(compressed, "zlib")
        assert decompressed.decode() == sample_text

    def test_invalid_algorithm(self, sample_text):
        """Test that invalid compression algorithms raise error."""
        with pytest.raises(CompressionError):
            compress_content(sample_text, "invalid")
        with pytest.raises(CompressionError):
            decompress_content(b"data", "invalid")


@pytest.mark.asyncio
class TestCompressedResource:
    async def test_read_write_with_compression(self, text_resource):
        """Test compressed resource read/write with compression enabled."""
        config = CompressionConfig(
            algorithm="gzip", min_size=0  # Force compression even for small content
        )
        compressed_resource = CompressedResource(
            uri=AnyUrl("memory://test/compressed"),
            wrapped_resource=text_resource,
            config=config,
            name="compressed_test",
        )

        # Write some content
        test_content = "Test content" * 100
        await compressed_resource.write(test_content)

        # Read it back
        content = await compressed_resource.read()
        assert content == test_content

    async def test_small_content_no_compression(self, text_resource):
        """Test that small content isn't compressed."""
        config = CompressionConfig(min_size=1000000)  # Set high threshold
        compressed_resource = CompressedResource(
            uri=AnyUrl("memory://test/compressed"),
            wrapped_resource=text_resource,
            config=config,
            name="compressed_test",
        )

        test_content = "Small content"
        await compressed_resource.write(test_content)
        assert compressed_resource.metadata.compression is None

        content = await compressed_resource.read()
        assert content == test_content


class TestTransformationUtils:
    def test_safe_encode_decode(self):
        """Test safe encoding and decoding utilities."""
        text = "Hello, World!"
        encoded = safe_encode(text)
        assert isinstance(encoded, bytes)
        decoded = safe_decode(encoded)
        assert decoded == text

    def test_text_transform(self):
        """Test text transformation wrapper."""
        uppercase = text_transform(str.upper)
        result = uppercase("hello")
        assert result == b"HELLO"

        result_bytes = uppercase(b"hello")
        assert result_bytes == b"HELLO"

    def test_bytes_transform(self):
        """Test bytes transformation wrapper."""
        reverse_bytes = bytes_transform(lambda x: bytes(reversed(x)))
        result = reverse_bytes(b"hello")
        assert result == b"olleh"

        result_str = reverse_bytes("hello")
        assert result_str == b"olleh"


@pytest.mark.asyncio
class TestTransformedResource:
    async def test_basic_transformation(self, text_resource):
        """Test basic resource transformation."""
        pipeline = TransformationPipeline(
            steps=[
                text_transform(str.upper),
                text_transform(lambda x: x.replace(" ", "_")),
            ],
            name="text_processor",
        )

        transformed = TransformedResource(
            uri=AnyUrl("memory://test/transformed"),
            wrapped_resource=text_resource,
            pipeline=pipeline,
            name="transformed_test",
        )

        content = await transformed.read()
        assert content == text_resource.text.upper().replace(" ", "_")

    async def test_json_pipeline(self, text_resource):
        """Test the JSON processing pipeline."""
        # Write JSON content to the source resource
        json_content = '{"key": "value", "number": 42}'
        await text_resource.write(json_content)

        pipeline = create_json_pipeline()
        transformed = TransformedResource(
            uri=AnyUrl("memory://test/json"),
            wrapped_resource=text_resource,
            pipeline=pipeline,
            name="json_test",
        )

        # The pipeline should format and compress the JSON
        compressed_content = await transformed.read()
        assert isinstance(compressed_content, bytes)  # Should be compressed

        # Decompress and verify the content
        decompressed = decompress_content(compressed_content, "gzip")
        assert json.loads(decompressed.decode()) == {"key": "value", "number": 42}

        # Write with reverse pipeline
        new_json = '{"new_key": "new_value"}'
        transformed.reverse_pipeline = True
        await transformed.write(new_json)

        # Read directly from source to verify
        original = await text_resource.read()
        parsed = json.loads(original)
        assert parsed == {"new_key": "new_value"}
