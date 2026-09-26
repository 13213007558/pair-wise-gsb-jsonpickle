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
-------------------
Large arrays are embedded as (optionally compressed) base64 text by
default, which inflates both the JSON document and memory usage. The
numpy extension also provides an explicit out-of-band buffer mode that
keeps binary payloads outside the JSON document::

    >>> import jsonpickle.ext.numpy as jsonpickle_numpy
    >>> jsonpickle_numpy.register_handlers()
    >>> json_text, buffers = jsonpickle_numpy.encode(value)
    >>> restored = jsonpickle_numpy.decode(json_text, buffers)

``encode()`` returns a ``(json_text, buffers)`` tuple: the JSON metadata
refers to payloads by index, and ``buffers`` is the ordered buffer table
those indexes refer to. Views of one underlying allocation share a
single buffer entry, and their sharing, dtype, offset, and strides
(including negative strides) are preserved across the roundtrip.

``decode()`` accepts each buffer as ``bytes``, ``bytearray``, or a
contiguous ``memoryview``. Non-contiguous memoryviews are explicitly
copied via ``memoryview.tobytes()``; other objects are rejected with a
``TypeError``. Writable buffers are used in place (zero-copy), while
read-only buffers are copied once so restored arrays are writeable,
matching the default mode. The buffer count, buffer lengths, and every
byte range reachable through the shape/strides/offset metadata are
validated before anything is handed to NumPy; out-of-bounds views raise
``ValueError``.

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
