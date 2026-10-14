"""Run: .venv/Scripts/python.exe -m unittest test_app -v (no API calls)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import app


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.lower().count(word)) for word in ("apple", "banana", "cherry")] + [0.01]


def write_pdf(path, pages):
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


class PDFWorkflowTest(unittest.TestCase):
    def test_multiple_isolation_single_and_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            app, "OpenAIEmbeddings", return_value=LocalEmbeddings()
        ):
            first, second, replacement, broken = [Path(folder) / name for name in
                                                  ("first.pdf", "second.pdf", "new.pdf", "broken.pdf")]
            write_pdf(first, ["apple orchard", "apple harvest"])
            write_pdf(second, ["banana plantation"])
            write_pdf(replacement, ["cherry garden"])
            broken.write_text("invalid PDF")
            db, message = app.build_db([str(first), str(second)], "test")
            self.assertIn("2개 파일 · 3페이지 · 3개 청크", message)
            for word, name in (("apple", first.name), ("banana", second.name)):
                doc = db.similarity_search(word, k=1)[0]
                self.assertEqual(doc.metadata["source_filename"], name)
                self.assertIn("page", doc.metadata)
                self.assertIn("source", doc.metadata)
            new_db, message = app.build_db([str(replacement)], "test")
            self.assertIn("1개 파일 · 1페이지 · 1개 청크", message)
            self.assertEqual(set(new_db.get()["documents"]), {"cherry garden"})
            self.assertNotEqual(db._collection.name, new_db._collection.name)
            db.delete_collection()
            new_db.delete_collection()
            failed_db, message = app.build_db([str(first), str(broken)], "test")
            self.assertIn("broken.pdf", message)
            self.assertIsNone(failed_db)
            # A failure while embedding the second file must remove partial chunks.
            original_add = app.Chroma.add_documents
            partial_collections = []
            def fail_second(db, docs, **kwargs):
                if docs[0].metadata["source_filename"] == second.name:
                    partial_collections.append(db)
                    raise RuntimeError("embedding failed")
                return original_add(db, docs, **kwargs)
            with patch.object(app.Chroma, "add_documents", fail_second):
                db, message = app.build_db([str(first), str(second)], "test")
            self.assertIsNone(db)
            self.assertIn("second.pdf", message)
            self.assertIsNone(partial_collections[0]._chroma_collection)
            self.assertIsNone(app.build_db([str(first)], "")[0])
            self.assertIsNone(app.build_db([], "test")[0])
            write_pdf(replacement, [""])
            self.assertIn("new.pdf", app.build_db([str(replacement)], "test")[1])
            write_pdf(replacement, ["cherry garden"])
            single_db, message = app.build_db(str(replacement), "test")
            self.assertIn("1개 파일", message)
            single_db.delete_collection()


if __name__ == "__main__":
    unittest.main()
