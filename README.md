# Airbnb Web Scrapping
Scrapping property information from Airbnb listings.

## Variables

| **Variable**                                                | **Data Type**    | **Description** |
|---------------------------------------------------------|-------------------|-----------------|
| Title                                                   | string   | Title of the listing |
| Summary                                                 | string  | Description of the listing |
| Price per night (CAD)                                  | int               | Price per night in Canadian Dollars |
| Sale price (CAD)                                       | int               | Sale price in Canadian Dollars |
| Number of guests                                      | int               | Number of guests the listing accommodates |
| Number of bedrooms                                    | int               | Number of bedrooms in the listing |
| Number of beds                                        | int               | Number of beds in the listing |
| Number of bathrooms                                   | float             | Number of bathrooms in the listing |
| Unavailable amenities shown                           | string (list)     | Amenities crossed out on the main page of the listing |
| Number of amenities                                   | int               | Total number of amenities available |
| Overall rating                                        | float             | Overall rating of the listing |
| Number of reviews                                     | int               | Number of reviews for the listing |
| Cleanliness                                           | float             | Cleanliness rating of the listing |
| Accuracy                                              | float             | Accuracy rating of the listing |
| Check-in                                              | float             | Check-in rating of the listing |
| Communication                                         | float             | Communication rating of the listing |
| Location                                              | float             | Location rating of the listing |
| Value                                                 | float             | Value rating of the listing |
| Title clean                                           | string (list)     | Processed text data from 'Title' column |
| downtown, condo, ..., sanctuary                        | binary            | '1' if word was used to describe the listing, '0' otherwise |
| Kitchen, Wifi, ..., Shared outdoor pool – available seasonally, open specific hours | binary | '1' if the amenity is available on the property (of those visible on the main page of the listing), '0' otherwise |
