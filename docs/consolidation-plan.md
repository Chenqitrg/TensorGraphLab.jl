# Consolidate the reviewed packages

The user requested one package and a readable explicit export list. Merge the reviewed node, graph-core and notation sources/tests into TensorGraphLab, retaining separate source files within a single module. Keep the public exported names and numerical behavior. Remove sibling package dependencies and example setup requirements; do not delete or modify the original repositories. Do not change the package UUID or version.

- Copy implementation files, remove former module wrappers and cross-package qualifiers, and list exports explicitly by responsibility.
- Bring all four test suites into this package. Preserve behavioral assertions; change imports/qualifiers only where consolidation requires it.
- Simplify Project.toml and examples/_setup.jl; document the unified source organization and independent setup.
- Verify the previous exported-name set, full test suite, all five standalone examples, repeated inclusion and a fresh isolated package directory without sibling repositories.
- Review the final diff and leave the changes available for review. No push or PR.
