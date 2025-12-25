"""
CHRONOS Data Pipeline

Connectors for real-time and batch data ingestion from cryptocurrency sources.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Standard transaction format for CHRONOS pipeline."""
    transaction_id: str
    timestamp: datetime
    features: List[float]  # 165 features
    source: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
            "source": self.source,
            "metadata": self.metadata or {}
        }


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    async def connect(self):
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to data source."""
        pass
    
    @abstractmethod
    async def stream_transactions(self) -> AsyncIterator[Transaction]:
        """Stream transactions in real-time."""
        pass
    
    @abstractmethod
    async def fetch_batch(self, start: datetime, end: datetime) -> List[Transaction]:
        """Fetch batch of historical transactions."""
        pass


class EllipticDataSource(DataSource):
    """
    Data source for Elliptic Bitcoin dataset.
    
    Used for development and testing.
    """
    
    def __init__(self, data_path: str = "data/elliptic"):
        self.data_path = data_path
        self.connected = False
    
    async def connect(self):
        logger.info(f"Connecting to Elliptic dataset at {self.data_path}")
        self.connected = True
    
    async def disconnect(self):
        logger.info("Disconnected from Elliptic dataset")
        self.connected = False
    
    async def stream_transactions(self) -> AsyncIterator[Transaction]:
        """Simulate streaming from Elliptic dataset."""
        if not self.connected:
            await self.connect()
        
        # In production, this would stream real transactions
        # For demo, yield placeholder
        yield Transaction(
            transaction_id="elliptic_demo",
            timestamp=datetime.now(),
            features=[0.0] * 165,
            source="elliptic"
        )
    
    async def fetch_batch(self, start: datetime, end: datetime) -> List[Transaction]:
        """Fetch batch from Elliptic dataset."""
        if not self.connected:
            await self.connect()
        return []


class BlockchainAPISource(DataSource):
    """
    Data source for real blockchain APIs.
    
    Supports:
    - Chainalysis API
    - Elliptic API
    - BlockCypher
    - Custom endpoints
    
    Note: Requires API credentials in environment variables.
    """
    
    def __init__(self, 
                 api_url: str,
                 api_key: str = None,
                 provider: str = "custom"):
        self.api_url = api_url
        self.api_key = api_key
        self.provider = provider
        self.connected = False
        self.session = None
    
    async def connect(self):
        """Connect to blockchain API."""
        import aiohttp
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.session = aiohttp.ClientSession(headers=headers)
        self.connected = True
        logger.info(f"Connected to {self.provider} API at {self.api_url}")
    
    async def disconnect(self):
        """Disconnect from API."""
        if self.session:
            await self.session.close()
        self.connected = False
        logger.info(f"Disconnected from {self.provider} API")
    
    async def stream_transactions(self) -> AsyncIterator[Transaction]:
        """
        Stream transactions from API.
        
        Implementation depends on provider's WebSocket/streaming support.
        """
        if not self.connected:
            await self.connect()
        
        # Placeholder for real implementation
        # Would connect to WebSocket endpoint for real-time data
        while True:
            # Poll or WebSocket receive
            await asyncio.sleep(1)
            # yield Transaction(...)
            break
    
    async def fetch_batch(self, start: datetime, end: datetime) -> List[Transaction]:
        """
        Fetch historical transactions from API.
        
        Parameters
        ----------
        start : datetime
            Start of time range
        end : datetime
            End of time range
            
        Returns
        -------
        List[Transaction]
            List of transactions in range
        """
        if not self.connected:
            await self.connect()
        
        if not self.session:
            return []
        
        # Example API call structure
        # async with self.session.get(
        #     f"{self.api_url}/transactions",
        #     params={"start": start.isoformat(), "end": end.isoformat()}
        # ) as response:
        #     data = await response.json()
        #     return [self._parse_transaction(tx) for tx in data]
        
        return []
    
    def _parse_transaction(self, data: Dict) -> Transaction:
        """Parse API response into Transaction."""
        # Implement based on provider's response format
        return Transaction(
            transaction_id=data.get("id", "unknown"),
            timestamp=datetime.fromisoformat(data.get("timestamp")),
            features=data.get("features", [0.0] * 165),
            source=self.provider,
            metadata=data
        )


class DataPipeline:
    """
    Main data pipeline orchestrator.
    
    Manages multiple data sources and routes transactions
    to the CHRONOS API for scoring.
    
    Usage:
        pipeline = DataPipeline()
        pipeline.add_source(EllipticDataSource())
        await pipeline.start()
    """
    
    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url
        self.sources: List[DataSource] = []
        self.running = False
        self.processed_count = 0
    
    def add_source(self, source: DataSource):
        """Add a data source to the pipeline."""
        self.sources.append(source)
        logger.info(f"Added data source: {source.__class__.__name__}")
    
    async def start(self):
        """Start the data pipeline."""
        logger.info("Starting CHRONOS data pipeline...")
        self.running = True
        
        # Connect all sources
        for source in self.sources:
            await source.connect()
        
        # Start streaming from all sources
        tasks = [
            self._process_source(source) 
            for source in self.sources
        ]
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the data pipeline."""
        logger.info("Stopping CHRONOS data pipeline...")
        self.running = False
        
        for source in self.sources:
            await source.disconnect()
    
    async def _process_source(self, source: DataSource):
        """Process transactions from a single source."""
        import aiohttp
        
        async for transaction in source.stream_transactions():
            if not self.running:
                break
            
            # Send to CHRONOS API for scoring
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.api_url}/predict",
                        json={
                            "transaction_id": transaction.transaction_id,
                            "features": transaction.features,
                            "return_explanation": False
                        }
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            self.processed_count += 1
                            logger.debug(
                                f"Processed {transaction.transaction_id}: "
                                f"risk={result.get('risk_score', 'N/A')}"
                            )
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")


# Export
__all__ = [
    "Transaction",
    "DataSource",
    "EllipticDataSource",
    "BlockchainAPISource",
    "DataPipeline"
]
