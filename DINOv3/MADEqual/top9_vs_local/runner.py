# Exported shared helpers; function bodies unchanged. See CODE_PROVENANCE.json.


def _new_output(output, inputs):
    for source in inputs:
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("output-dir must be separate from every input tree")
    # A failed or finished previous run must also remain untouched.
    output.mkdir(parents=True, exist_ok=False)
