### Intro to CNN for Image Processing 

#CNN_AHLAD_KUMAR_2020-09-11 22-11-27.png

<p align="center">
    <img src="https://github.com/DigitalCognition-GIS/sentinel_scihub_copernicus_eu/blob/master/Screen_captures/Sentinel_Screenshot%20from%202020-07-12%2011-56-48.png" width= "850px">
</p>

<h1 align="center">SENTINEL Experiments - </h1>

> This repository will contain both code and additional reading material refrences for analytics and pseudo-projects done with the Sentinel Data **- Sentinel 2020**
 
> If you are a GIS developer or a GeoSpatial Scientist - kindly feel free to contribute. 


<br/>


### Table of Contents of this repository

- [X] `A-- Intro to Sentinel...` 
- [X] `B-- Download and Preprocess Sentinel Data` 
- [X] `C-- Further explorations with Sentinel Data` 
- [X] `D-- Work in Progress` 
- [X] `Work in Progress` 
- [X] `Work in Progress` 
- [X] `Work in Progress` 


<br/>

### References - Always an ongoing effort - Work in Progress

### sentinel_scihub_copernicus_eu
Intial experiments with data from - sentinel_scihub_copernicus_eu

<br/>


- Source URL - SENTINEL -SAFE(Standard  Archive  Format  for Europe) - https://sentinel.esa.int/documents/247904/685211/Sentinel-2-Products-Specification-Document

> We get to notice the ```.safe``` file extension as seen for the - ```manifest.safe``` file . Also the same ```.SAFE``` extension is given for the actual Data file within the compressed ZIP File. The uncompressed Directory TREE structure is seen below :-   

<br/>


```
/Sentinel_data/S2A_MSIL1C_20160129T054102_N0201_R005_T43RFM_20160129T054903.SAFE$ tree

.
├── AUX_DATA
├── DATASTRIP
│   └── DS_MTI__20160129T092949_S20160129T054903
│       ├── MTD_DS.xml
│       └── QI_DATA
├── GRANULE
│   └── L1C_T43RFM_A003148_20160129T054903
│       ├── AUX_DATA
│       │   └── AUX_ECMWFT
│       ├── IMG_DATA
│       │   ├── T43RFM_20160129T054102_B01.jp2
│       │   ├── T43RFM_20160129T054102_B02.jp2
│       │   ├── T43RFM_20160129T054102_B03.jp2
│       │   ├── T43RFM_20160129T054102_B04.jp2
│       │   ├── T43RFM_20160129T054102_B05.jp2
│       │   ├── T43RFM_20160129T054102_B06.jp2
│       │   ├── T43RFM_20160129T054102_B07.jp2
│       │   ├── T43RFM_20160129T054102_B08.jp2
│       │   ├── T43RFM_20160129T054102_B09.jp2
│       │   ├── T43RFM_20160129T054102_B10.jp2
│       │   ├── T43RFM_20160129T054102_B11.jp2
│       │   ├── T43RFM_20160129T054102_B12.jp2
│       │   ├── T43RFM_20160129T054102_B8A.jp2
│       │   └── T43RFM_20160129T054102_TCI.jp2
│       ├── MTD_TL.xml
│       └── QI_DATA
│           ├── MSK_CLOUDS_B00.gml
│           ├── MSK_DEFECT_B01.gml
│           ├── MSK_DEFECT_B02.gml
│           ├── MSK_DEFECT_B03.gml
│           ├── MSK_DEFECT_B04.gml
│           ├── MSK_DEFECT_B05.gml
│           ├── MSK_DEFECT_B06.gml
│           ├── MSK_DEFECT_B07.gml
│           ├── MSK_DEFECT_B08.gml
│           ├── MSK_DEFECT_B09.gml
│           ├── MSK_DEFECT_B10.gml
│           ├── MSK_DEFECT_B11.gml
│           ├── MSK_DEFECT_B12.gml
│           ├── MSK_DEFECT_B8A.gml
│           ├── MSK_DETFOO_B01.gml
│           ├── MSK_DETFOO_B02.gml
│           ├── MSK_DETFOO_B03.gml
│           ├── MSK_DETFOO_B04.gml
│           ├── MSK_DETFOO_B05.gml
│           ├── MSK_DETFOO_B06.gml
│           ├── MSK_DETFOO_B07.gml
│           ├── MSK_DETFOO_B08.gml
│           ├── MSK_DETFOO_B09.gml
│           ├── MSK_DETFOO_B10.gml
│           ├── MSK_DETFOO_B11.gml
│           ├── MSK_DETFOO_B12.gml
│           ├── MSK_DETFOO_B8A.gml
│           ├── MSK_NODATA_B01.gml
│           ├── MSK_NODATA_B02.gml
│           ├── MSK_NODATA_B03.gml
│           ├── MSK_NODATA_B04.gml
│           ├── MSK_NODATA_B05.gml
│           ├── MSK_NODATA_B06.gml
│           ├── MSK_NODATA_B07.gml
│           ├── MSK_NODATA_B08.gml
│           ├── MSK_NODATA_B09.gml
│           ├── MSK_NODATA_B10.gml
│           ├── MSK_NODATA_B11.gml
│           ├── MSK_NODATA_B12.gml
│           ├── MSK_NODATA_B8A.gml
│           ├── MSK_SATURA_B01.gml
│           ├── MSK_SATURA_B02.gml
│           ├── MSK_SATURA_B03.gml
│           ├── MSK_SATURA_B04.gml
│           ├── MSK_SATURA_B05.gml
│           ├── MSK_SATURA_B06.gml
│           ├── MSK_SATURA_B07.gml
│           ├── MSK_SATURA_B08.gml
│           ├── MSK_SATURA_B09.gml
│           ├── MSK_SATURA_B10.gml
│           ├── MSK_SATURA_B11.gml
│           ├── MSK_SATURA_B12.gml
│           ├── MSK_SATURA_B8A.gml
│           ├── MSK_TECQUA_B01.gml
│           ├── MSK_TECQUA_B02.gml
│           ├── MSK_TECQUA_B03.gml
│           ├── MSK_TECQUA_B04.gml
│           ├── MSK_TECQUA_B05.gml
│           ├── MSK_TECQUA_B06.gml
│           ├── MSK_TECQUA_B07.gml
│           ├── MSK_TECQUA_B08.gml
│           ├── MSK_TECQUA_B09.gml
│           ├── MSK_TECQUA_B10.gml
│           ├── MSK_TECQUA_B11.gml
│           ├── MSK_TECQUA_B12.gml
│           ├── MSK_TECQUA_B8A.gml
│           └── T43RFM_20160129T054102_PVI.jp2
├── HTML
│   ├── banner_1.png
│   ├── banner_2.png
│   ├── banner_3.png
│   ├── star_bg.jpg
│   ├── UserProduct_index.html
│   └── UserProduct_index.xsl
├── INSPIRE.xml
├── manifest.safe
├── MTD_MSIL1C.xml
└── rep_info
    └── S2_User_Product_Level-1C_Metadata.xsd

11 directories, 94 files
```

<br/>

- Source URL - Sentinel-2-msi Data-Formats  - https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/data-formats 

<p align="center">
    <img src="https://github.com/DigitalCognition-GIS/sentinel_scihub_copernicus_eu/blob/master/Screen_captures/Screenshot%20from%202020-07-12%2013-08-26.png" width= "850px">
</p>


<br/>

- Source URL - https://gdal.org/drivers/vector/gml.html
- Source URL - https://gdal.org/drivers/vector/index.html
- Source URL - https://gdal.org/drivers/raster/eedai.html
- Source URL - Sentinel-2-msi Data-Formats  - https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/data-formats 

> The Sentinel Data has vector GML Files , these ```Geography Markup Language``` files are as listed within TREE structure of DIR = GRANULE > L1C_T43RFM_A003148_20160129T054903 > QI_DATA .    
The 'tagset' for this GML has Elements as listed below -    
- Within the outer Tagset - ```<eop:maskMembers><eop:maskMembers>```
- Within Tagset ```<eop:MaskFeature></eop:MaskFeature>``` we have ``` <eop:MaskFeature gml:id="OPAQUE.3">```
- Also within the Tagset ```<eop></eop>``` we have -  ```<eop:maskType codeSpace="urn:gs2:S2PDGS:maskType">CIRRUS</eop:maskType>```

<br/>

- Source - Further specifics read -- https://sentinel.esa.int/documents/247904/685211/Sentinel-2-Products-Specification-Document

> User Product Format - The  User  Product  is  formatted  by  default  as  a SENTINEL -SAFE(Standard  Archive  Format  for Europe) product.  

- Source URL - https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/definitions  

> Auxiliary Data - Describes all auxiliary information that will be used by the Payload Data Ground Segment (PDGS) for the processing of MSI data. The auxiliary data required in *MultiSpectral Instrument (MSI)* data generation are:

- Ground Image Processing Parameters (GIPP)  
- Digital Elevation Model (DEM) (see below)  
- Global Reference Image (GRI)  
- European Centre for Medium-Range Weather Forecasts (ECMWF): ozone, surface pressure and water vapour data required for Level-1C processing
- International Earth Rotation & Reference Systems service (IERS) data  
- Precise Orbit Determination (POD) data.  

> Datatake - The continuous acquisition of an image from one SENTINEL-2 satellite in a given MSI imaging mode is called a "datatake". The maximum length of an imaging datatake is 15,000 km (continuous observation from northern Russia to southern Africa).  

> Datastrip - Within a given *datatake*, a portion of image downlinked during a pass to a given station is termed a *"datastrip"*. If a particular orbit is acquired by more than one station, a datatake is composed of one or more datastrips. It is expected that the maximum length of a datastrip downlinked to a ground station is approximately 5,000 km.  

> DEM - Digital Elevation Model - *Orthorectification* in the L1C uses the 90m DEM (PlanetDEM 90). The PlanetDEM 90 is reprocessed from 90 metre SRTM (Shuttle Radar Topography Mission) source data. For PlanetDEM 90 data, the SRTM input data (v4.1) has been improved over specific mountain areas, and corrected over deserts and parts of the USA, using GDEM and NED.  

> Granules and Tiles - A granule is the *minimum indivisible partition of a product* (containing all possible spectral bands). For Level-0, Level-1A and Level-1B, granules are sub-images of a detector with a given number of lines along track.   
A granule covers approximately 25 km across-track and 23 km along-track.    
For Level-1C, the granules, also called tiles, are 100 km2 ortho-images in UTM/WGS84 projection.   


<br/>


- Source URL - https://eox.at/2015/12/understanding-sentinel-2-satellite-data/

> The best introduction to Sentinel satellite's and the Copernicus programme that i have read. 



<br/>

Rohit Dhankar - https://www.linkedin.com/in/rohitdhankar/




