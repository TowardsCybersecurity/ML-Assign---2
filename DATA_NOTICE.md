# Dataset provenance
Credit: AT&T Laboratories Cambridge, Olivetti Faces database.
The official scikit-learn documentation requests this acknowledgement:
https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset
The source archive was downloaded via the URL in scikit-learn 1.6.1 and verified
against its SHA256 b612fb967f2dc77c9c62d3e1266e0c73d5fca46a4b8906c18e454d41af987794.
Dataset arrays and archives are not redistributed in Git. The report contains only
the example faces needed for the assigned scientific analysis, with attribution.
The documentation gives attribution guidance, not an unrestricted redistribution
license; no claim of unrestricted licensing is made.

Local DNS failed for ndownloader.figshare.com and its S3 redirect. Curl with DNS
answers obtained from Google's public DNS resolved the download, with standard
TLS verification retained. A separate probml/pmtk3 mirror was downloaded only to
/tmp as a diagnostic and had the exact same archive SHA256. All experiments use
sklearn.datasets.fetch_olivetti_faces(shuffle=False) from the verified standard cache.
