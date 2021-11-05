This is a Python module in development.
It builds on the [nephosem](https://github.com/QLVL/nephosem/) package
and adds functions to create token-level vector space models in a semasiological workflow
like the one followed [here](https://cloudspotting.marianamontes.me/).

Next to the module there are two Jupyter notebooks:
`createClouds` is in Python and takes care of the creation of token-level distance matrices,
while `processClouds` is in R, uses the [`semcloud` package](https://github.com/montesmariana/semcloud)
and prepares the data to be used in the [nephovis tool](https://qlvl.github.io/NephoVis/) (see [here](https://github.com/QLVL/NephoVis)).

If you use the code here, please report success/failures in the [Issues](https://github.com/montesmariana/semasioFlow/issues).
