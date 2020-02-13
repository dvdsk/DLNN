## Quality Control
QC (quality control) is used to maintain working well formatted code without commen pitfalls. We automatically type check <sup>[1](#python typing)</sup>, format, lint and test any changes to the code base. 

We use the following tools to do this:

* mypy [static type checker]
* black [formatter]
* flake8 [linter]
* pytest [tester]

You can run the quality control on your own machine after installing the tools using:
```bash 
pip3 install black
pip3 install flake8
pip3 install mypy
pip3 install pytest
```

Then run them from the `src/` directory with 
```bash 
mypy main.py
black *
flake8 --ignore=E501,E302
pytest
```

<a name="myfootnote1">1</a>: python types can be statically checked similarly to when a compiler is used with C++ see: [python type checking](https://docs.python.org/3/library/typing.html) and [mypy](http://mypy-lang.org/)
