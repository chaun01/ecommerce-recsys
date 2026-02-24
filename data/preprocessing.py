"""
Data preprocessing pipeline for RetailRocket e-commerce dataset.

This module handles:
1. Loading and parsing raw CSV files
2. Event processing with implicit feedback weighting
3. Item feature extraction
4. Time-based train/validation/test splitting
5. User sequence generation
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetailRocketPreprocessor:
    """Preprocessor for RetailRocket dataset."""

    # Event weights for implicit feedback
    EVENT_WEIGHTS = {
        'view': 1.0,
        'addtocart': 2.0,
        'transaction': 3.0
    }

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        min_interactions: int = 5,
        max_sequence_length: int = 50
    ):
        """
        Initialize preprocessor.

        Args:
            input_dir: Directory containing raw CSV files
            output_dir: Directory to save processed files
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            min_interactions: Minimum interactions per user
            max_sequence_length: Maximum sequence length for user history
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.min_interactions = min_interactions
        self.max_sequence_length = max_sequence_length

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data holders
        self.events_df = None
        self.items_df = None
        self.categories_df = None

    def load_data(self):
        """Load raw CSV files."""
        logger.info("Loading raw data files...")

        # Load events
        events_path = self.input_dir / "events.csv"
        logger.info(f"Loading events from {events_path}")
        self.events_df = pd.read_csv(events_path)
        logger.info(f"Loaded {len(self.events_df):,} events")

        # Load item properties (may be split into multiple parts)
        item_files = list(self.input_dir.glob("item_properties*.csv"))
        logger.info(f"Loading item properties from {len(item_files)} file(s)")

        items_dfs = []
        chunk_size = 100000
        for item_file in item_files:
            logger.info(f"Reading {item_file.name} in chunks...")
            # Read in chunks to avoid memory error
            chunks = []
            for chunk in pd.read_csv(item_file, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            items_dfs.append(df)
            logger.info(f"Loaded {len(df):,} rows from {item_file.name}")

        self.items_df = pd.concat(items_dfs, ignore_index=True)
        logger.info(f"Loaded {len(self.items_df):,} item properties")

        # Load category tree
        categories_path = self.input_dir / "category_tree.csv"
        logger.info(f"Loading category tree from {categories_path}")
        self.categories_df = pd.read_csv(categories_path)
        logger.info(f"Loaded {len(self.categories_df):,} categories")

    def process_events(self) -> pd.DataFrame:
        """
        Process event data with timestamp conversion and weighting.

        Returns:
            Processed events DataFrame
        """
        logger.info("Processing events...")

        df = self.events_df.copy()

        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['date'] = df['datetime'].dt.date

        # Add event weights
        df['weight'] = df['event'].map(self.EVENT_WEIGHTS)

        # Remove rows with missing weights (invalid events)
        initial_count = len(df)
        df = df.dropna(subset=['weight'])
        logger.info(f"Removed {initial_count - len(df):,} invalid events")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Event distribution:")
        for event, count in df['event'].value_counts().items():
            logger.info(f"  {event}: {count:,} ({count/len(df)*100:.2f}%)")

        return df

    def filter_users_and_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter users and items with minimum interactions.

        Args:
            df: Events DataFrame

        Returns:
            Filtered DataFrame
        """
        logger.info(f"Filtering users and items with min {self.min_interactions} interactions...")

        initial_users = df['visitorid'].nunique()
        initial_items = df['itemid'].nunique()
        initial_interactions = len(df)

        # Iteratively filter until convergence
        prev_size = len(df)
        iteration = 0

        while True:
            iteration += 1

            # Filter users
            user_counts = df['visitorid'].value_counts()
            valid_users = user_counts[user_counts >= self.min_interactions].index
            df = df[df['visitorid'].isin(valid_users)]

            # Filter items
            item_counts = df['itemid'].value_counts()
            valid_items = item_counts[item_counts >= self.min_interactions].index
            df = df[df['itemid'].isin(valid_items)]

            # Check convergence
            if len(df) == prev_size:
                break
            prev_size = len(df)

            logger.info(f"  Iteration {iteration}: {len(df):,} interactions, "
                       f"{df['visitorid'].nunique():,} users, "
                       f"{df['itemid'].nunique():,} items")

        final_users = df['visitorid'].nunique()
        final_items = df['itemid'].nunique()
        final_interactions = len(df)

        logger.info(f"Filtering complete:")
        logger.info(f"  Users: {initial_users:,} → {final_users:,} "
                   f"({final_users/initial_users*100:.2f}%)")
        logger.info(f"  Items: {initial_items:,} → {final_items:,} "
                   f"({final_items/initial_items*100:.2f}%)")
        logger.info(f"  Interactions: {initial_interactions:,} → {final_interactions:,} "
                   f"({final_interactions/initial_interactions*100:.2f}%)")

        return df.reset_index(drop=True)

    def create_mappings(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        Create user and item ID mappings.

        Args:
            df: Events DataFrame

        Returns:
            Tuple of (user_mapping, item_mapping)
        """
        logger.info("Creating ID mappings...")

        # Create mappings from original IDs to sequential indices
        # Reserve index 0 for padding, start from 1
        unique_users = sorted(df['visitorid'].unique())
        unique_items = sorted(df['itemid'].unique())

        user_mapping = {user_id: idx + 1 for idx, user_id in enumerate(unique_users)}
        item_mapping = {item_id: idx + 1 for idx, item_id in enumerate(unique_items)}

        # Add mapped IDs to dataframe
        df['user_idx'] = df['visitorid'].map(user_mapping)
        df['item_idx'] = df['itemid'].map(item_mapping)

        logger.info(f"Created mappings: {len(user_mapping):,} users, {len(item_mapping):,} items")

        # Save mappings
        pd.DataFrame(list(user_mapping.items()), columns=['visitorid', 'user_idx']).to_parquet(
            self.output_dir / 'user_mapping.parquet'
        )
        pd.DataFrame(list(item_mapping.items()), columns=['itemid', 'item_idx']).to_parquet(
            self.output_dir / 'item_mapping.parquet'
        )

        return user_mapping, item_mapping

    def time_based_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform time-based train/validation/test split.

        Args:
            df: Events DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Performing time-based split...")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate split points
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # Split data
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(f"Split sizes:")
        logger.info(f"  Train: {len(train_df):,} ({len(train_df)/n*100:.2f}%)")
        logger.info(f"  Val:   {len(val_df):,} ({len(val_df)/n*100:.2f}%)")
        logger.info(f"  Test:  {len(test_df):,} ({len(test_df)/n*100:.2f}%)")

        # Log time ranges
        logger.info(f"Time ranges:")
        logger.info(f"  Train: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
        logger.info(f"  Val:   {val_df['datetime'].min()} to {val_df['datetime'].max()}")
        logger.info(f"  Test:  {test_df['datetime'].min()} to {test_df['datetime'].max()}")

        return train_df, val_df, test_df

    def extract_item_features(self, item_mapping: Dict) -> pd.DataFrame:
        """
        Extract and process item features.

        Args:
            item_mapping: Mapping from original item IDs to indices

        Returns:
            Item features DataFrame
        """
        logger.info("Extracting item features...")

        # Filter items that are in our mapping
        df = self.items_df[self.items_df['itemid'].isin(item_mapping.keys())].copy()

        # Add mapped item index
        df['item_idx'] = df['itemid'].map(item_mapping)

        # Pivot to get one row per item
        # Common properties: categoryid, available, price-related properties

        # Extract categoryid
        category_df = df[df['property'] == 'categoryid'][['item_idx', 'value']].copy()
        category_df.columns = ['item_idx', 'categoryid']
        category_df['categoryid'] = pd.to_numeric(category_df['categoryid'], errors='coerce')

        # Extract availability
        available_df = df[df['property'] == 'available'][['item_idx', 'value']].copy()
        available_df.columns = ['item_idx', 'available']
        available_df['available'] = pd.to_numeric(available_df['available'], errors='coerce')

        # Extract price information (property '790' seems to be price)
        price_df = df[df['property'] == '790'][['item_idx', 'value']].copy()
        price_df.columns = ['item_idx', 'price_raw']

        # Parse price (format: n*.000)
        def parse_price(val):
            try:
                if pd.isna(val):
                    return np.nan
                val_str = str(val)
                if val_str.startswith('n'):
                    return float(val_str[1:])
                return float(val_str)
            except:
                return np.nan

        price_df['price'] = price_df['price_raw'].apply(parse_price)
        price_df = price_df[['item_idx', 'price']]

        # Merge all features
        item_features = pd.DataFrame({'item_idx': range(len(item_mapping))})

        # Merge category
        item_features = item_features.merge(
            category_df.groupby('item_idx')['categoryid'].first().reset_index(),
            on='item_idx',
            how='left'
        )

        # Merge availability
        item_features = item_features.merge(
            available_df.groupby('item_idx')['available'].first().reset_index(),
            on='item_idx',
            how='left'
        )

        # Merge price
        item_features = item_features.merge(
            price_df.groupby('item_idx')['price'].first().reset_index(),
            on='item_idx',
            how='left'
        )

        # Fill missing values
        item_features['categoryid'] = item_features['categoryid'].fillna(-1).astype(int)
        item_features['available'] = item_features['available'].fillna(1).astype(int)
        item_features['price'] = item_features['price'].fillna(
            item_features['price'].median()
        )

        # Add price bucket
        item_features['price_bucket'] = pd.cut(
            item_features['price'],
            bins=[0, 1000, 5000, 10000, 50000, np.inf],
            labels=[0, 1, 2, 3, 4]
        )
        # Handle NaN in price_bucket (fill with median bucket)
        item_features['price_bucket'] = item_features['price_bucket'].cat.add_categories([-1])
        item_features['price_bucket'] = item_features['price_bucket'].fillna(-1).astype(int)

        logger.info(f"Extracted features for {len(item_features):,} items")
        logger.info(f"Feature coverage:")
        logger.info(f"  CategoryID: {(item_features['categoryid'] != -1).sum():,} "
                   f"({(item_features['categoryid'] != -1).sum()/len(item_features)*100:.2f}%)")
        logger.info(f"  Price: {item_features['price'].notna().sum():,} "
                   f"({item_features['price'].notna().sum()/len(item_features)*100:.2f}%)")

        return item_features

    def create_user_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user interaction sequences.

        Args:
            df: Events DataFrame

        Returns:
            User sequences DataFrame
        """
        logger.info("Creating user sequences...")

        sequences = []

        for user_idx, group in tqdm(df.groupby('user_idx'), desc="Processing users"):
            # Sort by timestamp
            group = group.sort_values('timestamp')

            # Get item sequence
            item_sequence = group['item_idx'].tolist()
            event_sequence = group['event'].tolist()
            weight_sequence = group['weight'].tolist()
            timestamp_sequence = group['timestamp'].tolist()

            sequences.append({
                'user_idx': user_idx,
                'item_sequence': item_sequence[-self.max_sequence_length:],
                'event_sequence': event_sequence[-self.max_sequence_length:],
                'weight_sequence': weight_sequence[-self.max_sequence_length:],
                'timestamp_sequence': timestamp_sequence[-self.max_sequence_length:],
                'sequence_length': len(item_sequence)
            })

        sequences_df = pd.DataFrame(sequences)

        logger.info(f"Created sequences for {len(sequences_df):,} users")
        logger.info(f"Average sequence length: {sequences_df['sequence_length'].mean():.2f}")
        logger.info(f"Median sequence length: {sequences_df['sequence_length'].median():.0f}")

        return sequences_df

    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        item_features: pd.DataFrame
    ):
        """Save processed data to parquet files."""
        logger.info("Saving processed data...")

        # Save splits
        train_df.to_parquet(self.output_dir / 'train_interactions.parquet', index=False)
        val_df.to_parquet(self.output_dir / 'val_interactions.parquet', index=False)
        test_df.to_parquet(self.output_dir / 'test_interactions.parquet', index=False)

        # Save item features
        item_features.to_parquet(self.output_dir / 'item_features.parquet', index=False)

        # Create user sequences for train set
        train_sequences = self.create_user_sequences(train_df)
        train_sequences.to_parquet(self.output_dir / 'train_sequences.parquet', index=False)

        logger.info(f"All processed data saved to {self.output_dir}")

        # Save statistics
        stats = {
            'num_users': train_df['user_idx'].nunique(),
            'num_items': train_df['item_idx'].nunique(),
            'num_train_interactions': len(train_df),
            'num_val_interactions': len(val_df),
            'num_test_interactions': len(test_df),
            'train_start': str(train_df['datetime'].min()),
            'train_end': str(train_df['datetime'].max()),
            'val_start': str(val_df['datetime'].min()),
            'val_end': str(val_df['datetime'].max()),
            'test_start': str(test_df['datetime'].min()),
            'test_end': str(test_df['datetime'].max()),
        }

        stats_df = pd.DataFrame([stats])
        stats_df.to_parquet(self.output_dir / 'dataset_stats.parquet', index=False)

        logger.info("Dataset statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    def run(self):
        """Run the complete preprocessing pipeline."""
        logger.info("="*80)
        logger.info("Starting RetailRocket Data Preprocessing Pipeline")
        logger.info("="*80)

        # Step 1: Load data
        self.load_data()

        # Step 2: Process events
        events_df = self.process_events()

        # Step 3: Filter users and items
        events_df = self.filter_users_and_items(events_df)

        # Step 4: Create mappings
        user_mapping, item_mapping = self.create_mappings(events_df)

        # Step 5: Time-based split
        train_df, val_df, test_df = self.time_based_split(events_df)

        # Step 6: Extract item features
        item_features = self.extract_item_features(item_mapping)

        # Step 7: Save processed data
        self.save_processed_data(train_df, val_df, test_df, item_features)

        logger.info("="*80)
        logger.info("Preprocessing Pipeline Complete!")
        logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Preprocess RetailRocket dataset for recommendation system'
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        default='data',
        help='Directory containing raw CSV files'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Directory to save processed files'
    )

    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Ratio of data for training (default: 0.7)'
    )

    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Ratio of data for validation (default: 0.15)'
    )

    parser.add_argument(
        '--min_interactions',
        type=int,
        default=5,
        help='Minimum interactions per user/item (default: 5)'
    )

    parser.add_argument(
        '--max_sequence_length',
        type=int,
        default=50,
        help='Maximum sequence length for user history (default: 50)'
    )

    args = parser.parse_args()

    # Create and run preprocessor
    preprocessor = RetailRocketPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_interactions=args.min_interactions,
        max_sequence_length=args.max_sequence_length
    )

    preprocessor.run()


if __name__ == '__main__':
    main()
