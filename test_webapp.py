#!/usr/bin/env python3
"""
Web Application Test Suite
Tests API endpoints and functionality
"""

import requests
import json
import time
import sys
from typing import Dict, List
import random

# API Configuration
API_BASE = "http://localhost:5000"
TIMEOUT = 10

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add_pass(self, test_name):
        self.passed += 1
        self.tests.append((test_name, True, None))
        print(f"{GREEN}✓{END} {test_name}")
    
    def add_fail(self, test_name, error):
        self.failed += 1
        self.tests.append((test_name, False, error))
        print(f"{RED}✗{END} {test_name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{BLUE}{'='*60}{END}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"{RED}{self.failed} tests failed{END}")
        else:
            print(f"{GREEN}All tests passed!{END}")
        print(f"{BLUE}{'='*60}{END}\n")
        
        return self.failed == 0


def test_health_check(results: TestResults):
    """Test health check endpoint"""
    try:
        response = requests.get(f"{API_BASE}/api/health", timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data['status'] == 'healthy', "Status not healthy"
        assert 'models_loaded' in data, "Missing models_loaded field"
        
        results.add_pass("Health Check")
        return data['models_loaded']
    except Exception as e:
        results.add_fail("Health Check", str(e))
        return False


def test_sample_users(results: TestResults) -> List[str]:
    """Test sample users endpoint"""
    try:
        response = requests.get(f"{API_BASE}/api/users/sample?limit=5", timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data['success'], "Request not successful"
        assert 'users' in data, "Missing users field"
        assert len(data['users']) > 0, "No users returned"
        
        results.add_pass(f"Sample Users (Got {len(data['users'])} users)")
        return data['users']
    except Exception as e:
        results.add_fail("Sample Users", str(e))
        return []


def test_recommendations(results: TestResults, user_id: str, top_n: int = 5):
    """Test recommendations endpoint"""
    try:
        payload = {
            "user_id": user_id,
            "top_n": top_n,
            "exclude_purchased": True
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/recommend",
            json=payload,
            timeout=TIMEOUT
        )
        response_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data['success'], "Request not successful"
        assert 'recommendations' in data, "Missing recommendations field"
        
        rec_count = len(data['recommendations'])
        results.add_pass(
            f"Recommendations for {user_id} "
            f"({rec_count} items, {response_time:.0f}ms)"
        )
        
        return data['recommendations']
    except Exception as e:
        results.add_fail(f"Recommendations for {user_id}", str(e))
        return []


def test_invalid_user(results: TestResults):
    """Test with invalid user ID"""
    try:
        payload = {
            "user_id": "INVALID_USER_12345",
            "top_n": 5
        }
        
        response = requests.post(
            f"{API_BASE}/api/recommend",
            json=payload,
            timeout=TIMEOUT
        )
        
        # Should return 200 but with no recommendations or error
        assert response.status_code in [200, 400], \
            f"Expected 200 or 400, got {response.status_code}"
        
        results.add_pass("Invalid User Handling")
    except Exception as e:
        results.add_fail("Invalid User Handling", str(e))


def test_missing_user_id(results: TestResults):
    """Test with missing user_id"""
    try:
        payload = {"top_n": 5}
        
        response = requests.post(
            f"{API_BASE}/api/recommend",
            json=payload,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        
        results.add_pass("Missing user_id Validation")
    except Exception as e:
        results.add_fail("Missing user_id Validation", str(e))


def test_invalid_top_n(results: TestResults):
    """Test with invalid top_n values"""
    try:
        # Test with top_n > 100
        payload = {
            "user_id": "test_user",
            "top_n": 150
        }
        
        response = requests.post(
            f"{API_BASE}/api/recommend",
            json=payload,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        
        results.add_pass("Invalid top_n Validation")
    except Exception as e:
        results.add_fail("Invalid top_n Validation", str(e))


def test_product_details(results: TestResults, product_id: str):
    """Test product details endpoint"""
    try:
        response = requests.get(
            f"{API_BASE}/api/product/{product_id}",
            timeout=TIMEOUT
        )
        
        # May be 200 (found) or 404 (not found)
        assert response.status_code in [200, 404], \
            f"Expected 200 or 404, got {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            assert data['success'], "Request not successful"
            assert 'product' in data, "Missing product field"
            results.add_pass(f"Product Details for {product_id}")
        else:
            results.add_pass(f"Product Details for {product_id} (not found)")
    except Exception as e:
        results.add_fail(f"Product Details for {product_id}", str(e))


def test_stats(results: TestResults):
    """Test stats endpoint"""
    try:
        response = requests.get(f"{API_BASE}/api/stats", timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data['success'], "Request not successful"
        assert 'stats' in data, "Missing stats field"
        
        stats = data['stats']
        results.add_pass(
            f"System Stats "
            f"({stats['total_users']} users, "
            f"{stats['total_products']} products)"
        )
    except Exception as e:
        results.add_fail("System Stats", str(e))


def run_tests():
    """Run all tests"""
    print(f"{BLUE}{'='*60}{END}")
    print(f"AI Recommendation System - API Test Suite")
    print(f"Testing: {API_BASE}")
    print(f"{BLUE}{'='*60}{END}\n")
    
    results = TestResults()
    
    # 1. Health check
    print(f"{YELLOW}1. Testing Health Check...{END}")
    models_ready = test_health_check(results)
    
    if not models_ready:
        print(f"{YELLOW}Waiting for models to load...{END}")
        for i in range(30):
            time.sleep(1)
            if test_health_check(results):
                models_ready = True
                break
    
    print()
    
    if not models_ready:
        print(f"{RED}Models not ready. Aborting tests.{END}")
        results.summary()
        return False
    
    # 2. Get sample users
    print(f"{YELLOW}2. Testing Sample Users...{END}")
    users = test_sample_users(results)
    print()
    
    if not users:
        print(f"{YELLOW}Using hardcoded test user...{END}")
        users = ["ATVPDKIKX0DER"]
    
    # 3. Test recommendations with multiple users
    print(f"{YELLOW}3. Testing Recommendations...{END}")
    recommendations = []
    for user_id in users[:3]:  # Test with first 3 users
        recs = test_recommendations(results, user_id, top_n=5)
        if recs:
            recommendations.extend(recs)
    print()
    
    # 4. Test error cases
    print(f"{YELLOW}4. Testing Error Handling...{END}")
    test_invalid_user(results)
    test_missing_user_id(results)
    test_invalid_top_n(results)
    print()
    
    # 5. Test product details
    if recommendations:
        print(f"{YELLOW}5. Testing Product Details...{END}")
        sample_products = random.sample(
            recommendations,
            min(3, len(recommendations))
        )
        for rec in sample_products:
            test_product_details(results, rec['product_id'])
        print()
    
    # 6. Test stats
    print(f"{YELLOW}6. Testing System Stats...{END}")
    test_stats(results)
    print()
    
    # Summary
    return results.summary()


if __name__ == "__main__":
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Tests interrupted by user{END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Fatal error: {e}{END}")
        sys.exit(1)
