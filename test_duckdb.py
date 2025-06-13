#!/usr/bin/env python3
"""
Quick test script to verify DuckDB contents
"""

import duckdb

def test_duckdb():
    conn = duckdb.connect('data/btc.duckdb')
    
    print('Table schema:')
    result = conn.execute('DESCRIBE btc_prices').fetchall()
    for row in result:
        print(f'  {row}')
    
    print('\nData sample:')
    result = conn.execute('SELECT * FROM btc_prices LIMIT 3').fetchall()
    for row in result:
        print(f'  {row}')
    
    print('\nData checks:')
    total_count = conn.execute('SELECT COUNT(*) FROM btc_prices').fetchone()[0]
    print(f'Total records: {total_count}')
    
    non_null_count = conn.execute('SELECT COUNT(*) FROM btc_prices WHERE close IS NOT NULL').fetchone()[0]
    print(f'Non-null close prices: {non_null_count}')
    
    date_range = conn.execute('SELECT MIN(date), MAX(date) FROM btc_prices').fetchone()
    print(f'Date range: {date_range[0]} to {date_range[1]}')
    
    conn.close()

if __name__ == "__main__":
    test_duckdb()
