------------------------------------------------------------------------------
How to install:
------------------------------------------------------------------------------

To install the repository, ensure that you enable your virtual environment. I 
can recommend py-venv. This can easily be installed on Unix-like OS (Mac, Linux).
For Mac, use **Brew**, and for Linux use the suitable package manager - i.e.
**pacman** for Arch-linux, **apt-get** for Ubuntu, etc.

Then create your virtual environment using:

```
python -m venv geoAI
```

Then activate your virtual environment using:

```
source geoAI/bin/activate
```

Thereafter, simply run:

```
pip install -r requirements.txt
```

to install all of the necessary dependencies.


------------------------------------------------------------------------------
How to run the CNNs:
------------------------------------------------------------------------------

To run any of the CNN models, please call the following command:

```
python src/<the_cnn_you_want_to_run.py>
```

where <the_cnn_you_want_to_run>={cnn.py, cnn_data_augment.py, cnn_binary.py}.


It should generate a directory called *results/*, wherein runs will be saved
according to your local time.

An example output of *src/cnn_data_augment.py*:


![](https://github.com/mengsig/GeoAI/blob/main/imgs/data_augment_result.png)

Which yields an MSE loss $L = 0.846$.
