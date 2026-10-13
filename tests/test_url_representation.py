import unittest

from training.url_representation import URLSurfaceView


class URLRepresentationTests(unittest.TestCase):
    def test_format_equivalence_preserves_destination_details(self):
        surface = "name:pass@Example.test:8443/Case%2FPath?a=1&a=2#Part"
        self.assertEqual(URLSurfaceView().fit_transform([
            surface, "http://" + surface, "HTTPS://" + surface
        ]), [surface] * 3)
        # A protocol-looking value in a path or query must remain intact.
        nested = "example.test/https://Other.test?next=http://Else.test"
        self.assertEqual(URLSurfaceView().transform([nested]), [nested])


if __name__ == "__main__":
    unittest.main()
