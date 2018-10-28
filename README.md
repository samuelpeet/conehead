# conehead :collision:

[![Build Status](https://travis-ci.com/samuelpeet/conehead.svg?branch=master)](https://travis-ci.com/samuelpeet/conehead) [![codecov](https://codecov.io/gh/samuelpeet/conehead/branch/master/graph/badge.svg)](https://codecov.io/gh/samuelpeet/conehead)

A collapsed-cone convolution radiotherapy dose calculation algorithm written in Python/Cython.

This project is currently in early experimental development. Expect the code to change dramatically over the coming months.

Inspiration has been drawn from [Cho et al (2012)](https://doi.org/10.3938/jkps.61.2073) and [Yang et al (2002)](https://doi.org/10.1118/1.1500767).

### Early development milestones:
* ~~Linac geometry~~ :heavy_check_mark:
* ~~Water phantom~~ :heavy_check_mark:
* ~~Point source~~ :heavy_check_mark:
* ~~Annular primary collimator source~~ :heavy_check_mark:
* ~~Exponential filter scatter source~~ :heavy_check_mark:
* ~~Voxel blocking (Jaws, MLC)~~ :heavy_check_mark:
* ~~Fluence calculation~~ :heavy_check_mark:
* ~~TERMA calculation~~ :heavy_check_mark:
* ~~Kernel calculated (EDKnrc)~~ :heavy_check_mark:
* ~~Kernel convolution~~ :heavy_check_mark:
* Kernel tilting
* ~~Horn tuning factor~~ :heavy_check_mark:
* ~~Square fields~~ :heavy_check_mark:
* Parse DICOM RT plan files (~~3DCRT~~ :heavy_check_mark:, IMRT/VMAT)
* Import CT images
* Support density overrides
* Option to optimise for out-of-field dosimetry
