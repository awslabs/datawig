# Datawig-JS

Datawig-JS is a Javascript port of [datawig](https://github.com/awslabs/datawig) `SimpleImputer`.

Datawig-JS enables human-in-the-loop quality control and missing value imputation. With this tool
it is possible to support missing value imputation use-cases entirely client side with a minimal
and intuitive user interface.

The port contains three main components:

- Support for Excel
- Client-side missing value imputation
  - live predictions
  - interpretability/explanations of learned model
- Active Learning for better label efficiency


**The current implementation is a prototype and should serve as a tech demo. It has several limitations including a relatively high memory consumption because the backend uses dense tensors. Sparse tensors are not yet supported by TensorFlow-JS releases.**



## How to run

Run `python server.py` and navigate with browser to [http://localhost:8081/datawig](http://localhost:8081/datawig).

To get started, feel free to use any of the Excel files in the `examples` directory.
