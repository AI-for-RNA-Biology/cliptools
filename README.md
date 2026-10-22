<!--
#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileCopyrightText: Copyright © 2026 University of Bern
#
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@idiap.ch>
# SPDX-FileContributor: Lisa Fournier <lisa.fournier@unibe.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
-->
# Project

This project aims at creating a package with multiple tools to analyze Clip-sequencing data.
For example, it will be useful to integrate Clip-sequencing data with other datasets (e.g a dataset of transcripts), to retrieve the transcripts that are cross-linked and where.

## Origin

cliptools was first developed at the [Idiap Research Institute](https://www.idiap.ch), where [version 0.1](https://github.com/idiap/cliptools) was released. It has been developed since at the University of Bern: this repository (version 0.2.0 and later) continues it.

## Installation

To install this library in another Python project, execute simply:

```bash
pip install git+https://github.com/AI-for-RNA-Biology/cliptools.git
```

or

```bash
git clone https://github.com/AI-for-RNA-Biology/cliptools.git
cd cliptools
pip install .
```

## Modules
- `processing`: regions around the PAS of 3' UTR isoforms, cross-linking of each region by a CLIP file, contingency
  tables and Fisher tests (cross-linking vs significant shifts), bootstrap distributions.
- `analysis`: load and combine the integrated CLIP results of several CLIP files (`load_and_combine_integrated_CLIP`).
