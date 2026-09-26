=====================
jsonpickle extensions
=====================

NumPy
-----
jsonpickle includes a built-in numpy extension. If would like to encode
sklearn models, numpy arrays, and other numpy-based data then you must
enable the numpy extension by registering its handlers::

    >>> import jsonpickle.ext.numpy as jsonpickle_numpy
    >>> jsonpickle_numpy.register_handlers()

See the example in ``examples/np_pd_extensions.py`` for a full demo of
how to use the extension.

Out-of-band buffers
~~~~~~~~~~~~~~~~~~~
For large arrays, the default base64 encoding inflates both the JSON
payload and memory usage. ``encode_oob`` returns a
``(json_metadata, buffers)`` pair instead: the JSON only references
binary payloads by index, and the raw buffers travel in an ordered
side-channel table. ``decode_oob`` takes the metadata plus the
caller-provided buffers (bytes, bytearray, or memoryview; non-contiguous
memoryviews are copied into packed bytes) and validates the buffer count
and every byte range reachable by a restored array or view before
handing anything to numpy::

    >>> metadata, buffers = jsonpickle_numpy.encode_oob(arr)
    >>> restored = jsonpickle_numpy.decode_oob(metadata, buffers)

Views sharing one underlying allocation produce a single buffer entry,
and sharing, dtypes, offsets, and strides are preserved exactly as in
the default mode.

Pandas
------
jsonpickle includes a built-in pandas extension. If you would like to
encode DataFrames, Series, or other pandas-specific instances then you
must enable the pandas extension by registering its handlers::

    >>> import jsonpickle.ext.pandas as jsonpickle_pandas
    >>> jsonpickle_pandas.register_handlers()

See the example in ``examples/np_pd_extensions.py`` for a full demo of
how to use the extension.

Ecdsa/gmpy2
-----
For the ecdsa module's keys, when trying to serialize them with
``gmpy2`` installed, jsonpickle will error unless the ``gmpy``
handlers are registered:

    >>> import jsonpickle.ext.gmpy as jsonpickle_gmpy
    >>> jsonpickle_gmpy.register_handlers()
