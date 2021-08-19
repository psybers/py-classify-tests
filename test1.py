@register_config('{{cookiecutter.project_name}}.test_pipeline_config')
class TestPipelineConfig(PipelineConfig):
    message: str = 'hello'

    def build(self, tmp_dir):
        return TestPipeline(self, tmp_dir)
