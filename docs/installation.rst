
.. _installation:

============
Installation
============

The installation is straight forward, you can use ``pip`` and clone or dowload a particular version from ``github`` repository.

Using ``pip``
-------------

You can install ``lucidmode``, and automatically all the dependencies, using pip::

    pip install lucidmode

Cloning from Github
-------------------

For the latest development version, first get the source from `Github`_::

    git clone https://github.com/lucidmode/lucidmode.git

Then navigate into the local ``lucidmode`` directory and if you run the following line it will install the package and all its dependencies::

    python setup.py install

Either option you choose, for the full use of the ``lucidmode`` package, you will need to have installed some depencies, all of them are listed in the ``requirements.txt`` file:

.. literalinclude:: ../requirements.txt
    :lines: 14-

*Those are just the lines with dependencies names and versions, you can check the full file* `Here`_

.. _Github: https://github.com/lucidmode
.. _Here: https://github.com/lucidmode/lucidmode/blob/main/requirements.txt
