"""
Index Mapper for PSX Indices
Dynamically maps stocks to indices based on market cap and volume
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

class PSXIndexMapper:
    """Maps stocks to PSX indices based on market data"""
    
    def __init__(self, live_df: pd.DataFrame = None, historical_df: pd.DataFrame = None):
        self.live_df = live_df
        self.historical_df = historical_df
        self.mapping = None
        
    def create_mapping(self, live_df: pd.DataFrame = None, historical_df: pd.DataFrame = None) -> pd.DataFrame:
        """Create index mapping based on available data"""
        
        if live_df is not None:
            self.live_df = live_df
        if historical_df is not None:
            self.historical_df = historical_df
        
        if self.live_df is None:
            return self._create_default_mapping()
        
        # Calculate market cap (price * volume approximation)
        if 'current' in self.live_df.columns and 'volume' in self.live_df.columns:
            self.live_df['market_cap_score'] = self.live_df['current'] * self.live_df['volume']
        else:
            self.live_df['market_cap_score'] = 1
        
        # Get all symbols
        all_symbols = self.live_df['symbol'].unique().tolist()
        
        # Sort by market cap score
        sorted_df = self.live_df.sort_values('market_cap_score', ascending=False)
        
        # Create index assignments
        n_stocks = len(all_symbols)
        
        mapping = []
        
        for i, symbol in enumerate(all_symbols):
            row = {
                'symbol': symbol,
                'KSE100': 1 if i < 100 else 0,  # Top 100 by market cap
                'KSE30': 1 if i < 30 else 0,    # Top 30 by market cap
                'KSEALL': 1,                     # All stocks in KSE-All Share
                'KMI30': 1 if i < 30 and self._is_shariah_compliant(symbol) else 0,
                'KMIALL': 1 if self._is_shariah_compliant(symbol) else 0,
                'PSXDIV20': 1 if self._is_high_dividend(symbol) else 0,
                'BKTI': 1 if self._is_banking_stock(symbol) else 0,
                'OGTI': 1 if self._is_oil_gas_stock(symbol) else 0,
            }
            mapping.append(row)
        
        self.mapping = pd.DataFrame(mapping)
        
        # Save mapping to file
        mapping_path = Path("data/indices/psx_index_mapping.csv")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        self.mapping.to_csv(mapping_path, index=False)
        
        print(f"✅ Created index mapping for {len(self.mapping)} stocks")
        print(f"   KSE-100: {self.mapping['KSE100'].sum()} stocks")
        print(f"   KSE-30: {self.mapping['KSE30'].sum()} stocks")
        print(f"   KMI-30: {self.mapping['KMI30'].sum()} stocks")
        print(f"   BKTI: {self.mapping['BKTI'].sum()} stocks")
        print(f"   OGTI: {self.mapping['OGTI'].sum()} stocks")
        
        return self.mapping
    
    def _is_shariah_compliant(self, symbol: str) -> bool:
        """Check if stock is Shariah compliant (simplified - based on sector)"""
        non_shariah_sectors = ['Banking', 'Insurance', 'Conventional Banking', 'Tobacco', 'Alcohol']
        
        # Get sector from historical data if available
        if self.historical_df is not None and 'sector' in self.historical_df.columns:
            stock_data = self.historical_df[self.historical_df['symbol'] == symbol]
            if len(stock_data) > 0:
                sector = stock_data.iloc[0].get('sector', '')
                if sector in non_shariah_sectors:
                    return False
        
        # Default to True for most stocks
        return True
    
    def _is_high_dividend(self, symbol: str) -> bool:
        """Check if stock typically pays high dividends"""
        # This would require actual dividend data
        # For now, use market cap as proxy
        if self.live_df is not None:
            stock_data = self.live_df[self.live_df['symbol'] == symbol]
            if len(stock_data) > 0:
                # Higher price stocks often pay higher dividends
                price = stock_data.iloc[0].get('current', 0)
                return price > 50  # Arbitrary threshold
        return False
    
    def _is_banking_stock(self, symbol: str) -> bool:
        """Check if stock is in banking sector"""
        banking_symbols = ['HBL', 'MCB', 'UBL', 'NBP', 'ABL', 'BAFL', 'BAHL', 'MEBL', 'BOP', 'FABL']
        return symbol in banking_symbols
    
    def _is_oil_gas_stock(self, symbol: str) -> bool:
        """Check if stock is in oil and gas sector"""
        oil_gas_symbols = ['OGDC', 'PPL', 'PSO', 'APL', 'ATRL', 'BYCO', 'MARI', 'SHEL', 'PRL', 'NRL']
        return symbol in oil_gas_symbols
    
    def _create_default_mapping(self) -> pd.DataFrame:
        """Create default mapping if no data available"""
        # Common PSX stocks with their index assignments
        default_mapping = [
            # KSE-100 constituents (top stocks)
            ('ENGRO', 1, 1, 1, 1, 1, 0, 0, 0),
            ('LUCK', 1, 1, 1, 1, 1, 0, 0, 0),
            ('HBL', 1, 1, 1, 1, 1, 0, 1, 0),
            ('MCB', 1, 1, 1, 1, 1, 0, 1, 0),
            ('UBL', 1, 0, 1, 1, 1, 0, 1, 0),
            ('NBP', 1, 0, 1, 0, 1, 0, 1, 0),
            ('OGDC', 1, 1, 1, 1, 1, 0, 0, 1),
            ('PPL', 1, 0, 1, 1, 1, 0, 0, 1),
            ('PSO', 1, 0, 1, 1, 1, 0, 0, 1),
            ('HUBC', 1, 1, 1, 1, 1, 0, 0, 0),
            ('KAPCO', 1, 0, 1, 0, 1, 0, 0, 0),
            ('FFC', 1, 1, 1, 1, 1, 0, 0, 0),
            ('FFBL', 1, 0, 1, 0, 1, 0, 0, 0),
            ('EFERT', 1, 0, 1, 1, 1, 0, 0, 0),
            ('NML', 1, 0, 1, 0, 1, 0, 0, 0),
            ('INDU', 1, 0, 1, 0, 1, 0, 0, 0),
            ('PSMC', 1, 0, 1, 0, 1, 0, 0, 0),
            ('SYS', 1, 0, 1, 0, 1, 0, 0, 0),
            ('MEBL', 1, 1, 1, 1, 1, 0, 1, 0),
            ('BAFL', 1, 0, 1, 1, 1, 0, 1, 0),
            ('ABL', 1, 0, 1, 0, 1, 0, 1, 0),
            ('BAHL', 1, 0, 1, 1, 1, 0, 1, 0),
            ('MARI', 1, 0, 1, 1, 1, 0, 0, 1),
            ('ATRL', 1, 0, 1, 0, 1, 0, 0, 1),
            ('APL', 1, 0, 1, 0, 1, 0, 0, 1),
            ('SHEL', 1, 0, 1, 0, 1, 0, 0, 1),
            ('DAWH', 1, 0, 1, 0, 1, 0, 0, 0),
            ('SNGP', 1, 0, 1, 0, 1, 0, 0, 0),
            ('SSGC', 1, 0, 1, 0, 1, 0, 0, 0),
            # KSE-All Share (all other stocks)
            ('FTSM', 0, 0, 1, 0, 0, 0, 0, 0),
            ('TSPL', 0, 0, 1, 0, 0, 0, 0, 0),
            ('KOHP', 0, 0, 1, 0, 0, 0, 0, 0),
            ('SIBL', 0, 0, 1, 0, 0, 0, 0, 0),
            ('TOWL', 0, 0, 1, 0, 0, 0, 0, 0),
            ('BAPL', 0, 0, 1, 0, 0, 0, 0, 0),
            ('DWTM', 0, 0, 1, 0, 0, 0, 0, 0),
            ('PACE', 0, 0, 1, 0, 0, 0, 0, 0),
            ('BELA', 0, 0, 1, 0, 0, 0, 0, 0),
            ('ASHT', 0, 0, 1, 0, 0, 0, 0, 0),
            ('ITANZ', 0, 0, 1, 0, 0, 0, 0, 0),
            ('ARCTM', 0, 0, 1, 0, 0, 0, 0, 0),
            ('IBLHL', 0, 0, 1, 0, 0, 0, 0, 0),
            ('SSOM', 0, 0, 1, 0, 0, 0, 0, 0),
            ('PMRS', 0, 0, 1, 0, 0, 0, 0, 0),
            ('ASTL', 0, 0, 1, 0, 0, 0, 0, 0),
            ('ISL', 0, 0, 1, 0, 0, 0, 0, 0),
            ('ASML', 0, 0, 1, 0, 0, 0, 0, 0),
            ('MUGHAL', 0, 0, 1, 0, 0, 0, 0, 0),
            ('TREET', 0, 0, 1, 0, 0, 0, 0, 0),
            ('PAEL', 0, 0, 1, 0, 0, 0, 0, 0),
            ('LOADS', 0, 0, 1, 0, 0, 0, 0, 0),
            ('PSEL', 0, 0, 1, 0, 0, 0, 0, 0),
            ('PNSC', 0, 0, 1, 0, 0, 0, 0, 0),
            ('GATM', 0, 0, 1, 0, 0, 0, 0, 0),
            ('ILP', 0, 0, 1, 0, 0, 0, 0, 0),
            ('HCAR', 0, 0, 1, 0, 0, 0, 0, 0),
            ('SAZEW', 0, 0, 1, 0, 0, 0, 0, 0),
            ('AVN', 0, 0, 1, 0, 0, 0, 0, 0),
            ('NETSOL', 0, 0, 1, 0, 0, 0, 0, 0),
            ('AIRLINK', 0, 0, 1, 0, 0, 0, 0, 0),
            ('COLG', 0, 0, 1, 0, 0, 0, 0, 0),
            ('UNILEVER', 0, 0, 1, 0, 0, 0, 0, 0),
            ('PAKT', 0, 0, 1, 0, 0, 0, 0, 0),
            ('PKGS', 0, 0, 1, 0, 0, 0, 0, 0),
            ('LOTCHEM', 0, 0, 1, 0, 0, 0, 0, 0),
            ('ICI', 0, 0, 1, 0, 0, 0, 0, 0),
        ]
        
        columns = ['symbol', 'KSE100', 'KSE30', 'KSEALL', 'KMI30', 'KMIALL', 'PSXDIV20', 'BKTI', 'OGTI']
        df = pd.DataFrame(default_mapping, columns=columns)
        
        # Ensure all symbols from live data are included
        if self.live_df is not None:
            all_symbols = set(self.live_df['symbol'].unique())
            existing_symbols = set(df['symbol'].unique())
            missing_symbols = all_symbols - existing_symbols
            
            for symbol in missing_symbols:
                df.loc[len(df)] = [symbol, 0, 0, 1, 0, 0, 0, 0, 0]
        
        return df


def get_index_mapping(live_df: pd.DataFrame = None, historical_df: pd.DataFrame = None) -> pd.DataFrame:
    """Get or create index mapping"""
    mapper = PSXIndexMapper(live_df, historical_df)
    return mapper.create_mapping(live_df, historical_df)


def filter_by_index(stocks_df: pd.DataFrame, index_code: str, mapping_df: pd.DataFrame = None) -> pd.DataFrame:
    """Filter stocks dataframe by index code"""
    if mapping_df is None:
        return stocks_df
    
    if index_code not in mapping_df.columns:
        return stocks_df
    
    eligible_symbols = mapping_df[mapping_df[index_code] == 1]['symbol'].tolist()
    
    if len(eligible_symbols) == 0:
        return stocks_df
    
    return stocks_df[stocks_df['symbol'].isin(eligible_symbols)]