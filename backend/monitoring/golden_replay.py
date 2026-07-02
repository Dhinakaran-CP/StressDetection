import json

class GoldenReplay:
    """
    Executes a regression replay on a 'golden' dataset to verify that the active 
    RuntimeEngine produces the exact expected outputs.
    """
    def __init__(self, runtime_engine):
        self.runtime_engine = runtime_engine
        
    def run_replay(self, golden_dataset_rows):
        """
        Takes a list of dicts (each containing face, voice, physio keys) 
        and runs them through the replay engine.
        Returns the prediction results.
        """
        if not self.runtime_engine.is_ready:
            return {"status": "error", "message": "Runtime engine not ready"}
            
        try:
            results = self.runtime_engine.replay(golden_dataset_rows)
            return {
                "status": "success",
                "count": len(results),
                "results": results
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Golden replay failed: {e}"
            }
