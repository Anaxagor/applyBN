# How to contribute

## Fast Checklist

- [ ] Fork the repository
- [ ] Clone the repository
- [ ] Create a new branch
- [ ] Ensure sklearn compatibility
- [ ] Write or update documentation with usage examples
- [ ] Check documentation with serve
- [ ] Write unittests
- [ ] Run tests to ensure everything works
- [ ] Commit your changes
- [ ] Push your branch
- [ ] Open a pull request

## Contribution Guide

We welcome contributions to the project. To contribute, please follow these steps:

1. **Cloning the repository**
    - Fork the repository to your GitHub account.
    - Clone your forked repository to your local machine using the following command:
      ```bash
      git clone https://github.com/Anaxagor/applyBN.git
      ```
    - Navigate to the project directory:
      ```bash
      cd applyBN
      ```

2. **Creating a branch**
    - Create a new branch for your feature or bug fix:
      ```bash
      git checkout -b your-branch-name
      ```

3. **Keeping everything sklearn-compatible**
    - Ensure that your contributions are compatible with `scikit-learn` where applicable. This includes following their conventions, inheriting their classes and ensuring interoperability.

4. **Writing documentation with usage example**
    - We use `mkdocs` for documentation. Add or update documentation in the `docs` directory.
    - Include usage examples to demonstrate how to use your feature or changes.
    - To preview the documentation locally, run:
      ```bash
      mkdocs serve
      ```

5. **Writing unittests**
    - Write unittests for your code to ensure its correctness.
    - Place your tests in the `tests` directory.
    - Run the tests to make sure everything works:
      ```bash
      pytest -s tests
      ```

6. **Submitting your changes**
    - Commit your changes with a descriptive commit message:
      ```bash
      git commit -m "Description of your changes"
      ```
    - Push your branch to your forked repository:
      ```bash
      git push origin your-branch-name
      ```
    - Open a pull request on the original repository and provide a detailed description of your changes.

Thank you for your contributions!
