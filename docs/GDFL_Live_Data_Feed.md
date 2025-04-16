# GDFL Live Data Feed Integration

## Overview

The GDFL Live Data Feed Integration is a critical component of the Enhanced Market Regime Optimizer pipeline that enables real-time market regime identification and strategy optimization. This integration replaces historical data sources with live streaming data from the GDFL (Global Data Feed Library) API, allowing the system to identify market regimes and optimize strategies in real-time.

The GDFL integration is designed to seamlessly connect with the existing pipeline components, providing a continuous stream of market data that can be processed by the market regime classifier, consolidator, and optimizer components.

## Purpose and Importance

The GDFL Live Data Feed Integration serves several important purposes:

1. **Real-time Market Regime Identification**: Enables the identification of market regimes in real-time, allowing traders to adapt their strategies to changing market conditions.

2. **Live Strategy Optimization**: Allows for the continuous optimization of trading strategies based on current market conditions.

3. **Immediate Feedback**: Provides immediate feedback on the performance of trading strategies in different market regimes.

4. **Automated Trading**: Facilitates automated trading by providing real-time market regime classifications and strategy recommendations.

5. **Continuous Learning**: Enables the system to continuously learn and adapt to changing market dynamics.

## Architecture

The GDFL Live Data Feed Integration follows a modular architecture that integrates with the existing pipeline components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  GDFL API       │────▶│  Data Processor │────▶│  Market Regime  │
│                 │     │                 │     │  Classifier     │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Strategy       │◀────│  Optimizer      │◀────│  Consolidator   │
│  Execution      │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Components

1. **GDFL API Client**: Connects to the GDFL API and retrieves real-time market data.
2. **Data Processor**: Processes the raw data from the GDFL API and converts it into the format expected by the market regime classifier.
3. **Market Regime Classifier**: Identifies the current market regime based on the processed data.
4. **Consolidator**: Combines the market regime classifications with strategy data.
5. **Optimizer**: Optimizes trading strategies based on the consolidated data.
6. **Strategy Execution**: Executes the optimized strategies in the market.

## GDFL API Integration

The GDFL API provides real-time market data through a RESTful API or WebSocket connection. The integration uses the following approach:

### Authentication

Authentication with the GDFL API is handled using credentials stored in the `GDFL_cred.txt` file. This file contains the API key and secret required to access the GDFL API.

```
api_key=your_api_key_here
api_secret=your_api_secret_here
api_endpoint=https://api.gdfl.com/v1
```

### Data Retrieval

The integration supports two methods for retrieving data from the GDFL API:

1. **Polling**: Periodically polls the GDFL API for new data.
2. **Streaming**: Establishes a WebSocket connection to receive real-time data as it becomes available.

The choice between polling and streaming depends on the specific requirements of the application and the capabilities of the GDFL API.

### Data Processing

The raw data from the GDFL API is processed to extract the relevant information needed for market regime identification:

1. **Price Data**: Open, high, low, close, and volume data for the underlying asset.
2. **Options Data**: Price, volume, open interest, and Greeks data for options contracts.
3. **Market Indicators**: Pre-calculated market indicators provided by the GDFL API.

The processed data is then passed to the market regime classifier for regime identification.

## Integration with Market Regime Classifier

The GDFL Live Data Feed Integration seamlessly integrates with the market regime classifier, providing real-time data for regime identification:

1. **Data Format Conversion**: Converts the GDFL data format to the format expected by the market regime classifier.
2. **Indicator Calculation**: Calculates additional indicators required by the market regime classifier that are not provided by the GDFL API.
3. **Regime Identification**: Passes the processed data to the market regime classifier for regime identification.

## Integration with Consolidator and Optimizer

The market regime classifications generated from the live data feed are integrated with the consolidator and optimizer components:

1. **Consolidation**: The market regime classifications are combined with strategy data in the consolidator.
2. **Optimization**: The consolidated data is used by the optimizer to identify the optimal strategies for the current market regime.
3. **Strategy Execution**: The optimized strategies are executed in the market.

## Configuration Options

The GDFL Live Data Feed Integration can be configured through several parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gdfl.api_endpoint` | string | 'https://api.gdfl.com/v1' | GDFL API endpoint URL |
| `gdfl.use_streaming` | boolean | true | Whether to use streaming (WebSocket) or polling (REST) |
| `gdfl.polling_interval` | int | 60 | Polling interval in seconds (for polling mode) |
| `gdfl.reconnect_attempts` | int | 3 | Number of reconnection attempts before giving up |
| `gdfl.reconnect_delay` | int | 5 | Delay between reconnection attempts in seconds |
| `gdfl.symbols` | list | ['NIFTY'] | List of symbols to retrieve data for |
| `gdfl.timeframes` | list | ['1m', '5m', '15m', '1h'] | List of timeframes to retrieve data for |
| `gdfl.include_options` | boolean | true | Whether to include options data |
| `gdfl.options_strikes` | int | 15 | Number of strikes around ATM to retrieve (7 above, 7 below, plus ATM) |
| `gdfl.options_expiries` | list | [0, 1, 2] | List of expiry indices to retrieve (0 = current expiry) |

Example configuration:

```python
config = {
    "gdfl": {
        "api_endpoint": "https://api.gdfl.com/v1",
        "use_streaming": True,
        "polling_interval": 60,
        "reconnect_attempts": 3,
        "reconnect_delay": 5,
        "symbols": ["NIFTY", "BANKNIFTY"],
        "timeframes": ["1m", "5m", "15m", "1h"],
        "include_options": True,
        "options_strikes": 15,
        "options_expiries": [0, 1, 2]
    }
}
```

## Implementation Guide

### Prerequisites

Before implementing the GDFL Live Data Feed Integration, ensure that you have:

1. **GDFL API Credentials**: Obtain API credentials from GDFL and store them in the `GDFL_cred.txt` file.
2. **Required Libraries**: Install the required libraries for connecting to the GDFL API (e.g., `requests`, `websocket-client`).
3. **Market Regime Classifier**: Ensure that the market regime classifier is properly configured and tested.

### Implementation Steps

1. **Create GDFL API Client**:
   - Implement a client for connecting to the GDFL API.
   - Handle authentication using the credentials from `GDFL_cred.txt`.
   - Implement methods for retrieving data through polling or streaming.

2. **Implement Data Processor**:
   - Create a processor for converting GDFL data to the format expected by the market regime classifier.
   - Implement methods for calculating additional indicators if needed.

3. **Integrate with Market Regime Classifier**:
   - Modify the market regime classifier to accept real-time data from the GDFL integration.
   - Ensure that the classifier can handle the frequency of data updates.

4. **Integrate with Consolidator and Optimizer**:
   - Modify the consolidator to handle real-time market regime classifications.
   - Adjust the optimizer to provide real-time strategy recommendations.

5. **Implement Error Handling and Reconnection Logic**:
   - Handle connection errors and implement reconnection logic.
   - Implement fallback mechanisms for when the GDFL API is unavailable.

6. **Test the Integration**:
   - Test the integration with sample data from the GDFL API.
   - Verify that the market regime classifications are accurate.
   - Ensure that the system can handle the volume of data from the GDFL API.

### Example Implementation

```python
import requests
import json
import time
import threading
import websocket
import logging
from datetime import datetime

class GDFLClient:
    """
    Client for connecting to the GDFL API.
    """
    
    def __init__(self, config):
        """
        Initialize the GDFL client.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.credentials = self._load_credentials()
        self.api_endpoint = self.credentials.get('api_endpoint', config.get('gdfl', {}).get('api_endpoint', 'https://api.gdfl.com/v1'))
        self.api_key = self.credentials.get('api_key')
        self.api_secret = self.credentials.get('api_secret')
        self.use_streaming = config.get('gdfl', {}).get('use_streaming', True)
        self.polling_interval = config.get('gdfl', {}).get('polling_interval', 60)
        self.reconnect_attempts = config.get('gdfl', {}).get('reconnect_attempts', 3)
        self.reconnect_delay = config.get('gdfl', {}).get('reconnect_delay', 5)
        self.symbols = config.get('gdfl', {}).get('symbols', ['NIFTY'])
        self.timeframes = config.get('gdfl', {}).get('timeframes', ['1m', '5m', '15m', '1h'])
        self.include_options = config.get('gdfl', {}).get('include_options', True)
        self.options_strikes = config.get('gdfl', {}).get('options_strikes', 15)
        self.options_expiries = config.get('gdfl', {}).get('options_expiries', [0, 1, 2])
        self.logger = logging.getLogger(__name__)
        self.ws = None
        self.running = False
        
    def _load_credentials(self):
        """
        Load GDFL API credentials from GDFL_cred.txt.
        
        Returns:
            dict: Dictionary containing API credentials
        """
        credentials = {}
        try:
            with open('GDFL_cred.txt', 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        credentials[key] = value
            return credentials
        except Exception as e:
            self.logger.error(f"Error loading GDFL credentials: {str(e)}")
            return {}
    
    def start(self):
        """
        Start the GDFL client.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.use_streaming:
            return self._start_streaming()
        else:
            return self._start_polling()
    
    def _start_streaming(self):
        """
        Start streaming data from the GDFL API.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.running = True
        threading.Thread(target=self._streaming_thread).start()
        return True
    
    def _streaming_thread(self):
        """
        Thread for streaming data from the GDFL API.
        """
        attempts = 0
        while self.running and attempts < self.reconnect_attempts:
            try:
                # Create WebSocket connection
                ws_url = f"{self.api_endpoint.replace('https://', 'wss://').replace('http://', 'ws://')}/stream"
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                self.ws.run_forever()
                
                # If we get here, the connection was closed
                if not self.running:
                    break
                
                # Increment attempts and wait before reconnecting
                attempts += 1
                self.logger.warning(f"WebSocket connection closed. Reconnecting in {self.reconnect_delay} seconds (attempt {attempts}/{self.reconnect_attempts})")
                time.sleep(self.reconnect_delay)
            except Exception as e:
                self.logger.error(f"Error in streaming thread: {str(e)}")
                attempts += 1
                time.sleep(self.reconnect_delay)
        
        if attempts >= self.reconnect_attempts:
            self.logger.error(f"Failed to connect to GDFL API after {self.reconnect_attempts} attempts")
    
    def _on_message(self, ws, message):
        """
        Handle WebSocket message.
        
        Args:
            ws: WebSocket connection
            message: Message received
        """
        try:
            data = json.loads(message)
            self._process_data(data)
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {str(e)}")
    
    def _on_error(self, ws, error):
        """
        Handle WebSocket error.
        
        Args:
            ws: WebSocket connection
            error: Error received
        """
        self.logger.error(f"WebSocket error: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """
        Handle WebSocket close.
        
        Args:
            ws: WebSocket connection
            close_status_code: Close status code
            close_msg: Close message
        """
        self.logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def _on_open(self, ws):
        """
        Handle WebSocket open.
        
        Args:
            ws: WebSocket connection
        """
        self.logger.info("WebSocket connection established")
        
        # Subscribe to data
        subscribe_message = {
            "action": "subscribe",
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "include_options": self.include_options,
            "options_strikes": self.options_strikes,
            "options_expiries": self.options_expiries,
            "api_key": self.api_key,
            "api_secret": self.api_secret
        }
        ws.send(json.dumps(subscribe_message))
    
    def _start_polling(self):
        """
        Start polling data from the GDFL API.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.running = True
        threading.Thread(target=self._polling_thread).start()
        return True
    
    def _polling_thread(self):
        """
        Thread for polling data from the GDFL API.
        """
        while self.running:
            try:
                # Poll for data
                data = self._poll_data()
                if data:
                    self._process_data(data)
                
                # Wait for next poll
                time.sleep(self.polling_interval)
            except Exception as e:
                self.logger.error(f"Error in polling thread: {str(e)}")
                time.sleep(self.reconnect_delay)
    
    def _poll_data(self):
        """
        Poll data from the GDFL API.
        
        Returns:
            dict: Data from the GDFL API
        """
        try:
            # Build request URL
            url = f"{self.api_endpoint}/data"
            
            # Build request parameters
            params = {
                "symbols": ",".join(self.symbols),
                "timeframes": ",".join(self.timeframes),
                "include_options": str(self.include_options).lower(),
                "options_strikes": self.options_strikes,
                "options_expiries": ",".join(map(str, self.options_expiries)),
                "api_key": self.api_key,
                "api_secret": self.api_secret
            }
            
            # Make request
            response = requests.get(url, params=params)
            
            # Check response
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Error polling data: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"Error polling data: {str(e)}")
            return None
    
    def _process_data(self, data):
        """
        Process data from the GDFL API.
        
        Args:
            data (dict): Data from the GDFL API
        """
        # Process data and pass to market regime classifier
        # This is a placeholder for the actual implementation
        pass
    
    def stop(self):
        """
        Stop the GDFL client.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.running = False
        if self.ws:
            self.ws.close()
        return True
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Connection errors

**Symptoms**: The GDFL integration fails to connect to the GDFL API.

**Solutions**:
- Check that the GDFL API credentials in `GDFL_cred.txt` are correct.
- Verify that the GDFL API endpoint is accessible from your network.
- Check for network issues that might prevent the connection.
- Ensure that the required libraries are installed.

#### Issue: Data processing errors

**Symptoms**: The GDFL integration fails to process data from the GDFL API.

**Solutions**:
- Check that the data format from the GDFL API matches the expected format.
- Verify that all required fields are present in the data.
- Ensure that the data processor is correctly implemented.
- Check for any changes in the GDFL API that might affect the data format.

#### Issue: Market regime classification errors

**Symptoms**: The market regime classifier fails to identify market regimes from the GDFL data.

**Solutions**:
- Check that the data format passed to the market regime classifier is correct.
- Verify that all required indicators are calculated correctly.
- Ensure that the market regime classifier is properly configured.
- Check for any changes in the market regime classifier that might affect the classification.

### Logging and Debugging

The GDFL integration includes comprehensive logging to help diagnose issues. By default, logs are written to the console and can be configured to write to a file.

To enable more detailed logging, you can adjust the logging level:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gdfl_debug.log"),
        logging.StreamHandler()
    ]
)
```

## Performance Considerations

### Optimizing for Speed

To optimize the GDFL integration for speed, consider the following:

1. **Use streaming**: Streaming provides real-time data with lower latency than polling.

2. **Optimize data processing**: Minimize the amount of processing required for each data update.

3. **Use efficient data structures**: Use memory-efficient data structures for storing and processing data.

4. **Parallelize processing**: Consider parallelizing the processing of different symbols or timeframes.

### Optimizing for Reliability

To optimize the GDFL integration for reliability, consider the following:

1. **Implement reconnection logic**: Automatically reconnect to the GDFL API if the connection is lost.

2. **Handle errors gracefully**: Catch and handle errors to prevent the integration from crashing.

3. **Implement fallback mechanisms**: Have fallback mechanisms for when the GDFL API is unavailable.

4. **Monitor connection status**: Continuously monitor the connection status and take action if issues are detected.

## Conclusion

The GDFL Live Data Feed Integration is a critical component of the Enhanced Market Regime Optimizer pipeline that enables real-time market regime identification and strategy optimization. By properly configuring and implementing the GDFL integration, you can replace historical data sources with live streaming data, allowing the system to identify market regimes and optimize strategies in real-time.

For more information on other components of the pipeline, refer to the following documentation:

- [Unified Market Regime Pipeline](Unified_Market_Regime_Pipeline.md)
- [Market Regime Formation](Market_Regime_Formation.md)
- [Consolidation](Consolidation.md)
- [Dimension Selection](Dimension_Selection.md)
- [Results Visualization](Results_Visualization.md)
- [PostgreSQL Integration](PostgreSQL_Integration.md)