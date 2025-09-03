## Development Guidelines
1. **Simplicity**: Keep the code simple. Every line should do one thing. Functions have a single responsibility and are concise.
2. **Readability**: Write self-explanatory code. Avoid unnecessary comments. If comments are needed, rewrite the code to make it clearer.
3. **Abstraction**: Build reusable utilities and separate business from data logic.
4. **Types**: Use type hints for inputs, prefer generic/wide types, avoid `Any`, and rely on inference where possible. Assume modern python versions (3.13+).
5. **Minimal changes**: Prefer making minimal changes to existing code. If larger changes are needed, consider staging them in smaller, incremental commits.
6. **Testability**: Write code that is easy to test. Avoid complex dependencies and side effects. Use dependency injection where appropriate.
7. **Documentation**: Never suggest docstrings except for tests. Always require docstrings for tests.

## Testing guidelines
1. **Integration**: Prefer larger integration tests over unit tests.
2. **Mocks**: Avoid using mocks unless absolutely necessary. Prefer using real objects and data.
3. **Less is more**: Write fewer tests that cover more functionality. Avoid writing tests for trivial code.
4. **Invariants**: Focus on testing invariants rather than specific implementations.
5. **Types**: Avoid testing types. Assume a type-checker is used.

## Miscellaneous
1. We have junior developers on the team. Don't always copy surrounding style. If a better way exists, use it.
2. Our team uses Jira for task management.
