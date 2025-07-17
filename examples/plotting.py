import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
import time
from pathlib import Path

def plot_optimization_iteration_costs(optimization_history: Dict[str, Any], config: Dict[str, Any]):
    """Plot optimization iteration costs and convergence."""
    if not optimization_history or 'solution' not in optimization_history:
        print("No optimization history available for plotting.")
        return
    
    solution = optimization_history['solution']
    
    # Create output directory for plots
    output_dir = Path(__file__).parent / "opt_plots"
    output_dir.mkdir(exist_ok=True)
    
    # Plot cost convergence with enhanced history
    plt.figure(figsize=(15, 10))
    
    # Cost history (from callback if available)
    if optimization_history.get('cost_history'):
        plt.subplot(2, 3, 1)
        plt.plot(optimization_history['cost_history'], 'b-', linewidth=2)
        plt.title('Cost Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.yscale('log')
    elif hasattr(solution, 'cost_history') and solution.cost_history is not None:
        plt.subplot(2, 3, 1)
        plt.plot(solution.cost_history, 'b-', linewidth=2)
        plt.title('Cost Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.yscale('log')
    
    # Gradient norm (from callback if available)
    if optimization_history.get('gradient_norm_history'):
        plt.subplot(2, 3, 2)
        plt.plot(optimization_history['gradient_norm_history'], 'r-', linewidth=2)
        plt.title('Gradient Norm')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.grid(True)
        plt.yscale('log')
    elif hasattr(solution, 'gradient_norm_history') and solution.gradient_norm_history is not None:
        plt.subplot(2, 3, 2)
        plt.plot(solution.gradient_norm_history, 'r-', linewidth=2)
        plt.title('Gradient Norm')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.grid(True)
        plt.yscale('log')
    
    # Step size
    if hasattr(solution, 'step_size_history') and solution.step_size_history is not None:
        plt.subplot(2, 3, 3)
        plt.plot(solution.step_size_history, 'g-', linewidth=2)
        plt.title('Step Size')
        plt.xlabel('Iteration')
        plt.ylabel('Step Size')
        plt.grid(True)
    
    # Trust region radius
    if hasattr(solution, 'trust_region_radius_history') and solution.trust_region_radius_history is not None:
        plt.subplot(2, 3, 4)
        plt.plot(solution.trust_region_radius_history, 'm-', linewidth=2)
        plt.title('Trust Region Radius')
        plt.xlabel('Iteration')
        plt.ylabel('Trust Region Radius')
        plt.grid(True)
    
    # Optimization statistics summary
    plt.subplot(2, 3, 5)
    plt.axis('off')
    
    stats_text = "Optimization Statistics\n\n"
    stats_text += f"Final Cost: {optimization_history.get('final_cost', 0.0):.6f}\n"
    stats_text += f"Total Iterations: {optimization_history.get('iterations', 0)}\n"
    stats_text += f"Converged: {optimization_history.get('converged', True)}\n"
    
    if optimization_history.get('cost_history'):
        initial_cost = optimization_history['cost_history'][0]
        final_cost = optimization_history['cost_history'][-1]
        cost_reduction = (initial_cost - final_cost) / initial_cost * 100
        stats_text += f"Cost Reduction: {cost_reduction:.2f}%\n"
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Convergence rate analysis
    if optimization_history.get('cost_history') and len(optimization_history['cost_history']) > 10:
        plt.subplot(2, 3, 6)
        costs = np.array(optimization_history['cost_history'])
        convergence_rate = np.abs(np.diff(np.log(costs)))
        plt.plot(convergence_rate, 'purple', linewidth=2)
        plt.title('Convergence Rate')
        plt.xlabel('Iteration')
        plt.ylabel('|Δ log(cost)|')
        plt.grid(True)
        plt.yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"optimization_convergence_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Optimization convergence plot saved to: {plot_path}")
    
    # Print optimization statistics
    print(f"\n=== Optimization Statistics ===")
    print(f"Final cost: {optimization_history.get('final_cost', 0.0):.6f}")
    print(f"Total iterations: {optimization_history.get('iterations', 0)}")
    print(f"Convergence status: {optimization_history.get('converged', True)}")   
    if optimization_history.get('cost_history'):
        initial_cost = optimization_history['cost_history'][0]
        final_cost = optimization_history['cost_history'][-1]
        cost_reduction = (initial_cost - final_cost) / initial_cost * 100
        print(f"Cost reduction: {cost_reduction:.2f}%")





