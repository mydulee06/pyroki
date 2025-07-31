#!/usr/bin/env python3
"""
Test client for trajectory optimization server
This script sends welding object position requests to the server
"""

import requests
import json
import numpy as np

def test_trajectory_optimization(x: float, y: float, z: float, yaw: float):
    """
    Send trajectory optimization request to server
    
    Args:
        x, y, z: welding object position in meters
        yaw: welding object yaw angle in radians
    """
    url = "http://localhost:8080/optimize_trajectory"
    
    data = {
        "x": x,
        "y": y, 
        "z": z,
        "yaw": yaw
    }
    
    print(f"Sending request to {url}")
    print(f"Welding object position: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={yaw:.3f} rad ({np.degrees(yaw):.1f} deg)")
    
    try:
        response = requests.post(url, json=data, timeout=300)  # 5 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Optimization Result ===")
            print(f"Success: {result['success']}")
            print(f"Message: {result['message']}")
            print(f"Max Position Error: {result['max_position_error']:.4f} m (tolerance: {result['position_tolerance']:.4f} m)")
            print(f"Max Orientation Error: {result['max_orientation_error']:.4f} rad (tolerance: {result['orientation_tolerance']:.4f} rad)")
            print(f"Max Collision Cost: {result['max_collision_cost']:.6f} (threshold: {result['collision_threshold']})")
            print(f"Number of timesteps: {result['num_timesteps']}")
            print(f"Trajectory shape: ({len(result['trajectory'])}, {len(result['trajectory'][0]) if result['trajectory'] else 0})")
            
            if not result['success']:
                print(f"Failed reasons: {', '.join(result['failed_reasons'])}")
            
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Make sure the server is running at localhost:8080")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Optimization took too long.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def check_server_health():
    """Check if server is running"""
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"Server status: {data.get('status', 'unknown')}")
            print(f"Message: {data.get('message', 'No message')}")
            return True
        else:
            print(f"Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("Server is not running or not reachable at localhost:8080")
        return False
    except Exception as e:
        print(f"Error checking server health: {str(e)}")
        return False

def main():
    print("=== Trajectory Optimization Client ===")
    
    # Check server health first
    print("Checking server health...")
    if not check_server_health():
        print("Please start the server first by running: python3 13_eetrack_simple.py")
        return
    
    print("\nServer is running! Ready to send optimization requests.\n")
    
    # Example test cases
    test_cases = [
        {"x": 0.2, "y": -0.3, "z": 0.0, "yaw": 0.0, "description": "Center position, no rotation"},
        {"x": 0.1, "y": -0.2, "z": 0.1, "yaw": np.pi/4, "description": "Slight offset with 45° rotation"},
        {"x": -0.1, "y": -0.4, "z": 0.2, "yaw": -np.pi/6, "description": "Negative X with -30° rotation"},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test Case {i}: {test_case['description']}")
        print(f"{'='*50}")
        
        result = test_trajectory_optimization(
            test_case["x"], 
            test_case["y"], 
            test_case["z"], 
            test_case["yaw"]
        )
        
        if result:
            # Optionally save result to file
            filename = f"trajectory_result_{i}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to {filename}")
        
        print("\nPress Enter to continue to next test case...")
        input()

def interactive_mode():
    """Interactive mode for manual testing"""
    print("=== Interactive Mode ===")
    print("Enter welding object positions manually")
    
    # Check server health first
    if not check_server_health():
        print("Please start the server first by running: python3 13_eetrack_simple.py")
        return
    
    while True:
        try:
            print("\nEnter welding object position (or 'quit' to exit):")
            x = input("X position (m): ")
            if x.lower() == 'quit':
                break
            x = float(x)
            
            y = float(input("Y position (m): "))
            z = float(input("Z position (m): "))
            yaw_deg = float(input("Yaw angle (degrees): "))
            yaw = np.radians(yaw_deg)
            
            result = test_trajectory_optimization(x, y, z, yaw)
            
            if result:
                save = input("\nSave result to file? (y/n): ")
                if save.lower() == 'y':
                    filename = f"trajectory_x{x:.1f}_y{y:.1f}_z{z:.1f}_yaw{yaw_deg:.1f}.json"
                    with open(filename, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"Result saved to {filename}")
            
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        main()