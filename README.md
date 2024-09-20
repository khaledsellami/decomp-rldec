<a name="readme-top"></a>

<h3 align="center">RLDec</h3>

  <p align="center">
    This project is an implementation of the RLDec decomposition approach as described in the paper "Extracting Microservices from Monolithic Systems using Deep Reinforcement Learning" <a href="TODO">(2024)</a>.

  </p>



<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#preparing-the-input-data">Preparing the input data</a></li>
        <li><a href="#training-the-model">Training the model</a></li>
        <li><a href="#decomposing-the-monolithic-application-using-the-trained-model">Decomposing the monolithic application using the trained model</a></li>
      </ul>
    </li>
    <li>
      <a href="#advanced-usage">Advanced usage</a>
      <ul>
        <li><a href="#using-the-analysis-and-parsing-tools-as-grpc-services">Using the analysis and parsing tools as gRPC services</a></li>
        <li><a href="#using-rldec-as-a-grpc-server">Using RLDec as a gRPC server</a></li>
        <li><a href="#using-rldec-as-a-python-module">Using RLDec as a Python module</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#version-history">Version History</a></li>
    <li><a href="#artifacts-package">Artifacts Package</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
# About The Project

This project is an implementation of the RLDec decomposition approach as described in the paper 
"Extracting Microservices from Monolithic Systems
using Deep Reinforcement Learning" [[1]](#1).

RLDec is a decomposition tool that analyzes the source code of a monolithic Java application and suggests the 
recommended microservices for each class in the system using a Deep Reinforcement Learning based method. 

The RLDec implementation contains the logic for generating the decomposition of the monolithic application. 
However, the analysis of the source code has to be done by another tool. This implementation is compatible with 
the packages [decomp-java-analysis-service](https://github.com/khaledsellami/decomp-java-analysis-service) and [decomp-parsing-service](https://github.com/khaledsellami/decomp-parsing-service.git) that handle the static analysis part of the process.
Otherwise, it is possible to use your own tool but the input to RLDec has to conform to the required types and structure.



# Getting Started

## Prerequisites

The main requirements are:
* Python 3.10 or higher

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/khaledsellami/decomp-rldec.git
   ```
2. Install rldec as a Python module:
   ```sh
   cd decomp-rldec/
   pip install -e .
   ```
   Or you can install only the required libraries:
   ```sh
    pip install -r requirements.txt
    ```



<!-- USAGE EXAMPLES -->
# Usage

## Preparing the input data
In order to use RLDec, the source code of the monolithic application has to be analyzed and parsed to extract the structural and semantic dependencies as described in the paper. You can use our anaysis and parsing tools or use your own tools to generate the required data:
### Using the analysis and parsing tools:
The details of how to use these tools and generate the analysis results with them can be found in their corresponding repositories: [decomp-java-analysis-service](https://github.com/khaledsellami/decomp-java-analysis-service) and [decomp-parsing-service](https://github.com/khaledsellami/decomp-parsing-service.git).

### Using a third-party analysis tool:

The folder that contains the static analysis results must contain the following strcuture:
```text
   path_to_my_data/
   └── name_of_the_app/
      ├── class_word_count.parquet: a NxM matrix representing the Bag-of-Word vectors of each class where N is the number of classes and M is the size of the vocabulary
      └── class_interactions.parquet: a NxM matrix representing the dependencies from each class to the others (calls, inheritance, field usage, etc) where N is the number of classes 
```


## Training the model
The training of the model can be done using the following command (example with the combsequential method):
```sh
   python main.py train your_app_name --data /path/to/data --output /path/to/save/model --config /path/to/config --approach combsequential --model model_name
```

The configuration file contains the hyperparameters of the approach. Examples of default configurations can be found in the [default_configs](rldec%2Fdefault_configs) folder.

You can also use the following command to get more details about the training:
```sh
   python main.py train --help
```

## Decomposing the monolithic application using the trained model
The decomposition of the monolithic application can be done using the following command:
```sh
   python main.py decompose your_app_name --model model_name --output /path/to/save/decomposition --path /path/to/saved/model
```
The output will be a JSON file that contains the decomposition of the monolithic application in addition to some metadata.


# Advanced usage
## Using the analysis and parsing tools as gRPC services
The analysis and parsing tools can be used as gRPC services. In this case, you do not need to prepare the data for RLDec. You can use the following command to train the model:
```sh
   python main.py train your_app_name --output /path/to/output --repo /path/or/github/link/to/source/code
``` 

## Using RLDec as a gRPC server
RLDec can be used as a gRPC server and its API can be consumed to train and generate the decompositions. You can start the server using the following command:
```sh
   python main.py start
```

For more details about the API, you can inspect the protobuf file "rldec.proto" in the [proto/rldec](protos%2Frldec) folder.

## Using RLDec as a Python module
RLDec can be used as a Python module and customized to fit your needs. For examples to how to use the module, you can inspect the files in the [jobs](jobs) folder.

<!-- ROADMAP -->
## Roadmap
* Improve the documentation of this module
* Integrate the CombFlattened approach into the CLI and Server
* Add an option to use the parsing service as a Python module
* Update the default configurations to include the new approaches paths

<!-- AUTHORS -->
## Authors

Khaled Sellami - [khaledsellami](https://github.com/khaledsellami) - khaled.sellami.1@ulaval.ca

<!-- VERSION -->
## Version History

* 0.5.2
    * Initial Public Release


<!-- Artifacts Package -->
## Artifacts Package

This project only contains the source code of the RLDec implementation. For reproducing the results of the paper, you can refer to the Artifacts Package in [Figshare](https://figshare.com/articles/software/RLDec_Artifacts_Package/24939159) or [its Github repository](https://github.com/khalsel/RLDec_EMSE_RP).


<!-- LICENSE -->
## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.



<!-- REFERENCES -->
## References

<a id="1">[1]</a> 
TODO add the reference to the paper when published


<p align="right">(<a href="#readme-top">back to top</a>)</p>

