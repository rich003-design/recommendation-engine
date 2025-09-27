# test_compile.py
from kfp import compiler
from pipelines.pipeline_definition import ml_pipeline

if __name__ == "__main__":
    try:
        compiler.Compiler().compile(
            pipeline_func=ml_pipeline,
            package_path='pipeline.json'
        )
        print("✅ Pipeline compiled successfully to pipeline.json")
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
