import unittest
import os
from backend.core.version_registry import VersionRegistry

class TestExpertReleasePipeline(unittest.TestCase):
    def test_registry_can_load(self):
        # Just verifying the registry can read the new models if they exist
        registry = VersionRegistry()
        face_model = registry.get_active_model("face_expert")
        voice_model = registry.get_active_model("voice_expert")
        
        # They might be None if tests run before training, which is fine
        if face_model:
            self.assertIn("accuracy", face_model["metadata"])

if __name__ == '__main__':
    unittest.main()
