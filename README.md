# ![descrive](web/static/img/DescriveFullLogo.png)
`descrive` is a [web app](http://liblo.ng/descrive) where users can discover how products on Kijiji are described. Give `descrive` a search string, and it will scrape Kijiji for matching listings (if it hasn't already done so in the past), then run an NLP-based pipeline to extract the most commonly described features and the most common corresponding descriptors.

[debian_setup_example.sh](debian_setup_example.sh) contains a set of commands that can be used to set up and run the web app on a Debian-based distro. Don't be surprised if one or more steps fails, though. I wouldn't recommend using an instance with less than 3 GB RAM for this app.

Credit to [Sam Chow](https://www.linkedin.com/in/chowsam/) for the logo design.

## Problem statement

The classifieds market in Canada generates $500M in revenue annually. Sellers are bad at writing ads, particularly in choosing what features to describe and how to describe them, and in organizing this information. Buyers are presented with only a few broad categories of products and services in which to search, and the rest of their query must be in the form of a search string.

## Solution

`descrive` extracts features and feature descriptors from text descriptions of products. Features, and the descriptors for each, are ranked by multiplicity, and presented to the seller. This provides the seller with valuable insights on how to describe their product most clearly. The groundwork is laid for an interface through which buyers could intelligently filter through classifieds ads based on their features and descriptors. Improving the UX in this way would drive more traffic to the classifieds site, ultimately increasing company revenue.
