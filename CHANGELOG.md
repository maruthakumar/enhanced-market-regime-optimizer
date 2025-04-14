# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository setup
- Consolidated Greek sentiment analysis implementation
  - Support for delta range from 0.5 to 0.01
  - Expiry weightage: 70% near expiry, 30% next expiry
- Consolidated trending OI with PA implementation
  - Analysis of ATM plus 7 strikes above and below (15 total)
  - Rolling calculation for trending OI of calls and puts
- Consolidated market regime classifier implementation
  - Support for all 18 regime formations
  - Dynamic weighting of indicators
- Enhanced PostgreSQL integration for 1-minute rolling storage
  - Time-series optimized tables and indexes
  - Data retention policies
- Market regime naming standardization
  - Enumeration to prevent typos
  - Validation and conversion functions

### Fixed
- Corrected "voltatile" typo to "volatile" throughout the codebase
- Removed duplicate implementations:
  - Consolidated 4 Greek sentiment analysis implementations into 1
  - Consolidated 2 trending OI with PA implementations into 1
  - Consolidated 2 market regime formation implementations into 1
