Highlevel API
=============

On top of the base API this library already implements some useful classes:

- An :py:class:`ArraySource`: Source for numpy arrays with slicing functionaliy. And a :py:class:`SoundFileSource`: Source for sound files using pysoundfile.
- Commonly known spectral, temporal and other Features (e.g. ``RootMeanSquare`` or ``SpectralFlatness`` etc.)
- A :py:class:`DefaultDictSink`:  Sink receiving results in dictionary.


Source
------

.. automodule:: sigfeat.source.array
  :members:


.. automodule:: sigfeat.source.soundfile
  :members:


Feature
-------

.. automodule:: sigfeat.feature
  :members:


Extractor
---------

.. automodule:: sigfeat.extractor
  :members:


Sink
----

.. automodule:: sigfeat.sink
  :members:


Preprocess
----------

.. automodule:: sigfeat.preprocess
  :members:


Base level API
==============

Contains the abstract base classes.

Source
------

.. automodule:: sigfeat.base.source
  :members:


Feature
-------

.. automodule:: sigfeat.base.feature
  :members:


Result
------

  .. automodule:: sigfeat.base.result
    :members:


Sink
----

.. automodule:: sigfeat.base.sink
  :members:


Preprocess
----------

.. automodule:: sigfeat.base.preprocess
  :members:



Parameter
---------

.. automodule:: sigfeat.base.parameter
  :members:



Metadata
--------

.. automodule:: sigfeat.base.metadata
  :members:
