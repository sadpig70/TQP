# Contributing to TQP

Thank you for your interest in contributing to TQP!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/sadpig70/TQP.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run tests: `cargo test --workspace`
6. Verify code quality: `cargo clippy --workspace -- -D warnings`
7. Submit a pull request

## Code Style

- Follow Rust conventions (use `cargo fmt`)
- Ensure no warnings in `cargo clippy`
- Add documentation for public APIs
- Write tests for new features
- Keep commits atomic and well-described

## Areas for Contribution

- **Algorithms**: New VQE ansätze, QAOA improvements
- **Hardware**: AWS Braket, Google Cirq integrations
- **Performance**: SIMD optimizations, GPU support
- **Documentation**: Examples, tutorials

## Reporting Issues

Please include:

- TQP version
- Rust version (`rustc --version`)
- Minimal reproduction code
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
