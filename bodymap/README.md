Body Map
========

Anatomical maps for plotting in Python and JavaScript.

Contents
--------

This repository contains a collection of "maps" that can be used for plotting
anatomical regions.

Each map consists of:

- An SVG image with each body region traced as a closed path and labeled with a
  unique, human-readable name.
- A YAML tree of text labels for each body region, arranged hierarchically.

The top level of this repo contains a Python script to load the SVG and YAML
files and assign colors to each labeled region.

Goals (WIP):

- A Python library to load a map and vocabulary by name and generate a
  colorized plot, similar to a choropleth for a geographical map.
- A JavaScript module and scaffolding to do the same, using D3, both in-browser
  and through node.
- More detailed anatomical maps and structured vocabularies for plotting
  medical, especially dermatological, data.
