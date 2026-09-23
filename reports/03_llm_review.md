# LLM second opinion on uncertain pairs

Model: `qwen2.5:7b-instruct` via Ollama, temperature 0, JSON-only output validated against a schema (bool / 0..1 float / string), one retry, then fall back to the classifier. Uncertain band: 0.3 < p < 0.7 of `random_forest`. 134 pairs, 134 valid answers, 816s wall time (cached answers are free).

## Whole test set
| setting | precision | recall | f1 |
|---|---|---|---|
| classifier only | 0.8516 | 0.8 | 0.825 |
| classifier + LLM in uncertain band | 0.906 | 0.8182 | 0.8599 |

## Inside the uncertain band
| decider (uncertain band only) | precision | recall | f1 |
|---|---|---|---|
| classifier at tuned threshold | 0.6 | 0.6957 | 0.6443 |
| LLM qwen2.5:7b-instruct | 0.7941 | 0.7826 | 0.7883 |

## Where the LLM fixed the classifier
- label=0, clf p=0.58, LLM=False (0.2): `Escort Passport Radar And Laser Detector - Black Finish - 8500` vs `Escort Passport 9500ix Radar/Laser Detector`  
  reason: Different model numbers and one listing provides a description while the other does not.
- label=1, clf p=0.42, LLM=True (0.95): `Onkyo 6 Disc CD Player Changer - DXC390B` vs `ONKYO 6-DISC CD CHANGER W/VLSC, BLACK NIC - DX-C390`  
  reason: Both listings describe a Onkyo 6-disc CD changer with a black finish.
- label=0, clf p=0.64, LLM=False (0.7): `Weber Performer 22-1/2' Charcoal Grill - 848001` vs `Weber 841001 Performer 22.5' Charcoal Grill in Black`  
  reason: The color variant is different (Blue vs Black).
- label=0, clf p=0.50, LLM=False (0.6): `Canon High Capacity Color Ink Cartridge - Color Ink - CL51` vs `Canon PG-50 High Capacity Black Ink Cartridge For PIXMA MP450 Printer - 0616B002`  
  reason: The color ink cartridge in Listing A is not the same as the black ink cartridge in Listing B.
- label=1, clf p=0.37, LLM=True (0.95): `Canon Photo Ink Cartridge - CL52` vs `Canon CL-52 Photo Ink Cartridge For PIXMA iP6210D and PIXMA iP6220D Printers - 0619B002`  
  reason: Both listings refer to the Canon CL52 photo ink cartridge for the same printer models.
- label=1, clf p=0.34, LLM=True (0.95): `Panasonic All-in-One Flatbed Laser Fax - KXFLB801` vs `Panasonic KX-FLB801 Multifunction Printer - KX-FLB801`  
  reason: Both listings describe a Panasonic KXFLB801 All-in-One Flatbed Laser Fax with similar features.

## Where the LLM was wrong
- label=0, clf p=0.47, LLM=True (0.95): `Canon Black Ink Cartridge - Black - PG40BK` vs `Canon PG-40 Black Ink Cartridge - PG-40-BK-TWIN`  
  reason: Both listings describe a Canon black ink cartridge with the same model identifier.
- label=1, clf p=0.54, LLM=False (0.3): `Tech Craft Avalon Series TV Stand - Black Finish - ABS32` vs `Techcraft Veneto Series ABS32 TV Stand`  
  reason: Different series and price discrepancy suggests different products.
- label=1, clf p=0.67, LLM=False (0.3): `Samsung DLP TV Stand In Black - TR72BX` vs `Samsung TR72B TV Stand`  
  reason: The price and description differ significantly between the two listings.
- label=0, clf p=0.42, LLM=True (0.95): `LG DLE5955W White XL Capacity Electric Dryer - DLE5955WH` vs `LG XL Capacity Electric Dryer`  
  reason: Both listings refer to an LG electric dryer with XL capacity, though details vary slightly in description.
- label=1, clf p=0.66, LLM=False (0.6): `Audiovox Xpress XM Satellite Radio FM Direct Adapter - XMFM1` vs `Audiovox XMFM1 Switch Box`  
  reason: While both products are related to Audiovox XM Satellite Radio, they have different names and prices, indicating they might be different variants or models.
- label=1, clf p=0.54, LLM=False (0.7): `LaCie 1TB FireWire 800/FireWire 400/USB 2.0 External Hard Drive - 301199U` vs `LaCie Big Disk Extreme+ Hard Drive - 301199U`  
  reason: While both are LaCie 1TB external hard drives with multiple interfaces, the specific model names differ.
