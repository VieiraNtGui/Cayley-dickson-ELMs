# Cayley-Dickson ELMs

Extreme Learning Machines were proposed as a quick-to-train alternative to the Multilayer Perceptron (MLP). Originally dubbed Random-Vector Functional Links (RvFLs) these models are trained by means of a direct method, adjusting weights by a least squares problem (LSP). In order to use higher-dimension algebras we define the equivalence between a real LSP and a LSP in such algebras. In this repository you can find our works that implement ELMs based on hypercomplex-valued algebras, as well as a particular work on Cayley-Dickson algebras. 

For more detailed information, please refer to the associated papers:

###  _Extreme Learning Machines on Cayley-Dickson Algebra Applied for Color Image Auto-Encoding_, Vieira and Valle, 2020, DOI: 10.1109/IJCNN48605.2020.9207495
This paper aims to provide a useful framework for extreme learning machines (ELMs) on Cayley-Dickson algebras. Cayley-Dickson algebras, which include complex numbers, quaternions, and octonions as particular instances, are hyper-complex algebras defined using a recursive procedure. Firstly, we review some basic concepts on Cayley-Dickson algebras and formulate Cayley-Dickson matrix product using real-valued linear algebra. Then, we propose the Cayley-Dickson ELMs and derive their learning using Cayley-Dickson least squares problem. Lastly, we compare the performance of real-valued and four-dimensional Cayley-Dickson ELM models, including quaternion-valued ELM, in an experiment on color image auto-encoding using the well-known CIFAR dataset.

**Authors:**
* *Marcos Eduardo Valle and Guilherme Vieira - University of Campinas*

### _A general framework for hypercomplex-valued extreme learning machines_, Vieira and Valle, 2022, DOI: 10.1016/j.jcmds.2022.100032
This paper aims to establish a framework for extreme learning machines (ELMs) on general hypercomplex algebras. Hypercomplex neural networks are machine learning models that feature higher-dimension numbers as parameters, inputs, and outputs. Firstly, we review broad hypercomplex algebras and show a framework to operate in these algebras through real-valued linear algebra operations in a robust manner. We proceed to explore a handful of well-known four-dimensional examples. Then, we propose the hypercomplex-valued ELMs and derive their learning using a hypercomplex-valued least-squares problem. Finally, we compare real and hypercomplex-valued ELM models’ performance in an experiment on time-series prediction and another on color image auto-encoding. The computational experiments highlight the excellent performance of hypercomplex-valued ELMs to treat multi-dimensional data, including models based on unusual hypercomplex algebras.

**Authors:**
* *Marcos Eduardo Valle and Guilherme Vieira - University of Campinas*
