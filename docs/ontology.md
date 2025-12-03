# Class Ontology Documentation

## Overview

This document describes the class mapping and ontology for the MTSD road sign detection system.

## Class Categories

MTSD road signs are organized into several categories:

### 1. Regulatory Signs
Signs that impose legal requirements or restrictions.
- Examples: `regulatory--stop--g1`, `regulatory--no-entry--g1`, `regulatory--maximum-speed-limit-45--g3`

### 2. Warning Signs
Signs that warn of potential hazards or conditions.
- Examples: `warning--pedestrians-crossing--g10`, `warning--railroad-crossing--g1`

### 3. Information Signs
Signs that provide information about facilities, services, or locations.
- Examples: `information--end-of-built-up-area--g1`, `information--tram-bus-stop--g2`

### 4. Complementary Signs
Signs that modify or supplement other signs.
- Examples: `complementary--maximum-speed-limit-15--g1`

## Class Naming Convention

MTSD uses a hierarchical naming convention:
```
{category}--{description}--{variant}
```

Example: `regulatory--maximum-speed-limit-45--g3`
- Category: `regulatory`
- Description: `maximum-speed-limit-45`
- Variant: `g3` (group 3)

## Class Mapping

The full class list is stored in:
- `/Users/weixianfu/Documents/Datas/mtsd/classes.json`
- Used by `configs/data.yaml` for training

## Future Considerations

- Mapping to canonical EU road sign categories
- Cross-dataset compatibility (e.g., GTSDB)
- Multi-language support for class names

