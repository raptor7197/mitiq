# Resource Requirements for Quantum Error Mitigation with Mitiq

This guide provides a comprehensive analysis of the computational resources required for implementing various quantum error mitigation (QEM) techniques in Mitiq. Understanding these requirements is crucial for efficient implementation and cost-effective quantum computing.

```{figure} ../img/resource_requirements_workflow.svg
---
width: 700px
name: resource_requirements_workflow
---
The resource requirements workflow in Mitiq is fully explained in the sections below.
```

## 1. Understanding Resource Requirements

When implementing QEM techniques, it's essential to understand their resource overhead. While direct cost in dollars is challenging to measure programmatically (as it varies by provider and isn't always exposed via APIs), we can use circuit-level metrics to estimate the computational cost. These metrics are particularly important for:

- **Budget planning and resource allocation**
- **Performance optimization**
- **Hardware selection and configuration**
- **Scaling considerations**

## 2. Circuit Count Analysis

Each QEM technique requires a different number of circuit executions, which directly impacts the total runtime and cost.

### Zero-Noise Extrapolation (ZNE)
```python
import cirq
import numpy as np
from mitiq import zne, pec, cdr
from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.benchmarks import generate_rb_circuits
from cirq import DensityMatrixSimulator, depolarize
from mitiq.interface import convert_to_mitiq

# ZNE
circuit = generate_rb_circuits(n_qubits=1, num_cliffords=2, return_type="cirq")[0]
print(f"Original circuit:\n{circuit}")

def execute(circuit, noise_level=0.01):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    executed with depolarizing noise.
    """
    mitiq_circuit, _ = convert_to_mitiq(circuit)
    noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real

noisy_value = execute(circuit)
ideal_value = execute(circuit, noise_level=0.0)
print(f"Ideal value: {ideal_value:.5f}")
print(f"Noisy value: {noisy_value:.5f}")
print(f"Error without mitigation: {abs(ideal_value - noisy_value):.5f}")

print("\n--- Method 1: One-step ZNE ---")
mitigated_result = zne.execute_with_zne(circuit, execute)
print(f"Mitigated value: {mitigated_result:.5f}")
print(f"Error with mitigation (ZNE): {abs(ideal_value - mitigated_result):.5f}")

print("\n--- Method 2: ZNE with custom factory ---")
linear_fac = LinearFactory(scale_factors=[1.0, 2.0])
mitigated_result_custom = zne.execute_with_zne(
    circuit, execute, factory=linear_fac, scale_noise=fold_gates_at_random
)
print(f"Mitigated value (custom): {mitigated_result_custom:.5f}")
print(f"Error with custom ZNE: {abs(ideal_value - mitigated_result_custom):.5f}")

print("\n--- Method 3: Two-stage ZNE ---")
scale_factors = [1.0, 2.0, 3.0]

folded_circuits = zne.construct_circuits(
    circuit=circuit,
    scale_factors=scale_factors,
    scale_method=fold_gates_at_random  
)

print(f"Number of folded circuits: {len(folded_circuits)}")
for i, circ in enumerate(folded_circuits):
    print(f"Circuit {i+1} (scale factor {scale_factors[i]}):")
    print(f"  Number of gates: {len(list(circ.all_operations()))}")

results = [execute(circuit) for circuit in folded_circuits]
print(f"Execution results: {results}")

extrapolation_method = RichardsonFactory(scale_factors=scale_factors).extrapolate
two_stage_zne_result = zne.combine_results(
    scale_factors, results, extrapolation_method
)

print(f"Two-stage ZNE result: {two_stage_zne_result:.5f}")
print(f"Error with two-stage ZNE: {abs(ideal_value - two_stage_zne_result):.5f}")

print("\n=== Probabilistic Error Cancellation (PEC) ===")

pec_circuit = cirq.Circuit(
    cirq.H(cirq.LineQubit(0)),
    cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1))
)

try:
    print("PEC requires detailed noise model setup - skipping for this example")
    print("Refer to Mitiq documentation for complete PEC implementation")
except Exception as e:
    print(f"PEC construction failed: {e}")

print("\n=== Clifford Data Regression (CDR) ===")

training_circuits = generate_rb_circuits(n_qubits=2, num_cliffords=20, trials=5, return_type="cirq")
target_circuit = generate_rb_circuits(n_qubits=2, num_cliffords=30, return_type="cirq")[0]

try:
    cdr_circuits = cdr.construct_circuits(
        target_circuit,
        training_circuits=training_circuits
    )
    
    print(f"Number of CDR circuits: {len(cdr_circuits)}")
    print(f"Training circuits: {len(training_circuits)}")
    print(f"Target circuit depth: {len(target_circuit)}")
    
except Exception as e:
    print(f"CDR construction failed: {e}")

print("\n=== Summary ===")
print("ZNE successfully demonstrated with three methods:")
print("1. One-step execute_with_zne()")
print("2. Custom factory with execute_with_zne()")
print("3. Two-stage construct_circuits() -> combine_results()")
print("\nKey correction: construct_circuits() uses 'scale_method', not 'factory'")
```
## 3. Gate Overhead Analysis

The introduction of additional gates is a critical consideration, especially for 2-qubit gates, which are typically the noisiest operations on quantum hardware.

### Example: Analyzing Gate Overhead
```python
def analyze_gate_overhead(circuit):
    operations = list(circuit.all_operations())  # Convert generator to list
    total_gates = len(operations)
    two_qubit_gates = len([op for op in operations if len(op.qubits) == 2])
    single_qubit_gates = total_gates - two_qubit_gates

    print(f"Total gates: {total_gates}")
    print(f"2-qubit gates: {two_qubit_gates}")
    print(f"1-qubit gates: {single_qubit_gates}")
    if total_gates > 0:
        print(f"2-qubit gate ratio: {two_qubit_gates/total_gates:.2%}")
    else:
        print("No gates in the circuit.")

circuit = generate_rb_circuits(n_qubits=2, num_cliffords=50)[0]
analyze_gate_overhead(circuit)
```

Mitiq provides a sophisticated two-stage approach:

1. **Circuit Construction:** Using `mitiq.xyz.construct_circuits`
   - Generates all necessary circuits for the QEM technique
   - Allows pre-execution analysis and resource estimation

2. **Result Combination:** Using `mitiq.xyz.combine_results`
   - Processes the results from all circuits
   - Applies the appropriate error mitigation
   - Produces the final, mitigated result

```python
import cirq
import numpy as np
from mitiq import zne
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.inference import LinearFactory
from mitiq.benchmarks import generate_rb_circuits
from cirq import DensityMatrixSimulator, depolarize
from mitiq.interface import convert_to_mitiq

print(" Stage 1: Circuit Construction")
circuit = generate_rb_circuits(n_qubits=2, num_cliffords=50, return_type="cirq")[0]
print(f"Original circuit depth: {len(circuit)}")

scale_factors = [1.0, 2.0, 3.0]

zne_circuits = zne.construct_circuits(
    circuit,
    scale_factors=scale_factors,
    scale_method=fold_gates_at_random  
)

print(f"Number of ZNE circuits generated: {len(zne_circuits)}")
for i, circ in enumerate(zne_circuits):
    print(f"Circuit {i+1} (scale factor {scale_factors[i]}): {len(circ)} operations")

print("\n Stage 2: Circuit Execution ")

def execute_circuit(circuit, noise_level=0.01):
    mitiq_circuit, _ = convert_to_mitiq(circuit)
    noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
    
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    
    ground_state_prob = rho[0, 0].real
    return ground_state_prob

results = [execute_circuit(circ) for circ in zne_circuits]
print(f"Execution results: {[f'{r:.4f}' for r in results]}")


linear_factory = LinearFactory(scale_factors=scale_factors)
mitigated_result = zne.combine_results(
    scale_factors,           # scale_factors first
    results,                 # results second  
    linear_factory.extrapolate  # extrapolation method
)

print(f"Mitigated result: {mitigated_result:.6f}")

unmitigated_result = execute_circuit(circuit)
ideal_result = execute_circuit(circuit, noise_level=0.0)

print(f"\n Comparison between the mitigated and the unmitigated result  ")
print(f"Ideal result (no noise): {ideal_result:.6f}")
print(f"Unmitigated result: {unmitigated_result:.6f}")
print(f"Mitigated result (ZNE): {mitigated_result:.6f}")
print(f"Error without mitigation: {abs(ideal_result - unmitigated_result):.6f}")
print(f"Error with ZNE mitigation: {abs(ideal_result - mitigated_result):.6f}")

one_step_result = zne.execute_with_zne(
    circuit, 
    execute_circuit,
    factory=LinearFactory(scale_factors=[1.0, 2.0, 3.0]),
    scale_noise=fold_gates_at_random
)
print(f"One-step ZNE result: {one_step_result:.6f}")
print(f"Error with one-step ZNE: {abs(ideal_result - one_step_result):.6f}")
```

### Example: Benchmarking Different Techniques
```python
import time
import cirq
import numpy as np
from mitiq import zne, pec, cdr
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.inference import LinearFactory
from mitiq.benchmarks import generate_rb_circuits
from cirq import DensityMatrixSimulator, depolarize
from mitiq.interface import convert_to_mitiq

def execute_circuit(circuit, noise_level=0.01):
    mitiq_circuit, _ = convert_to_mitiq(circuit)
    noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real

def benchmark_zne(circuit, scale_factors=[1.0, 2.0, 3.0]):
    start_time = time.time()
    
    circuits = zne.construct_circuits(circuit, scale_factors=scale_factors, scale_method=fold_gates_at_random)
    results = [execute_circuit(circ) for circ in circuits]
    factory = LinearFactory(scale_factors=scale_factors)
    mitigated_result = zne.combine_results(scale_factors, results, factory.extrapolate)
    
    return {
        'technique': 'ZNE',
        'time': time.time() - start_time,
        'circuits': len(circuits),
        'result': mitigated_result
    }

def benchmark_pec(circuit, num_samples=20):
    start_time = time.time()
    try:
        from mitiq.pec.representations.depolarizing import represent_operation_with_local_depolarizing_noise
        gate_types = {type(op.gate) for op in circuit.all_operations()}
        representations = {gt: represent_operation_with_local_depolarizing_noise for gt in gate_types}
        
        circuits = pec.construct_circuits(circuit, representations=representations, num_samples=num_samples)
        results = [execute_circuit(circ) for circ in circuits]
        mitigated_result = np.mean(results)
        
        return {'technique': 'PEC', 'time': time.time() - start_time, 'circuits': len(circuits), 'result': mitigated_result}
    except Exception as e:
        return {'technique': 'PEC', 'time': time.time() - start_time, 'circuits': 0, 'result': None, 'error': str(e)}

def benchmark_cdr(circuit, training_circuits):
    start_time = time.time()
    try:
        circuits = cdr.construct_circuits(circuit, training_circuits=training_circuits)
        results = [execute_circuit(circ) for circ in circuits]
        mitigated_result = results[0] if results else 0.5
        
        return {'technique': 'CDR', 'time': time.time() - start_time, 'circuits': len(circuits), 'result': mitigated_result}
    except Exception as e:
        return {'technique': 'CDR', 'time': time.time() - start_time, 'circuits': 0, 'result': None, 'error': str(e)}

circuit = generate_rb_circuits(n_qubits=2, num_cliffords=50, return_type="cirq")[0]
training_circuits = generate_rb_circuits(n_qubits=2, num_cliffords=20, trials=3, return_type="cirq")
ideal_result = execute_circuit(circuit, noise_level=0.0)
noisy_result = execute_circuit(circuit, noise_level=0.01)

print("QEM Benchmark Results ")
print(f"Ideal: {ideal_result:.4f}, Noisy: {noisy_result:.4f}, Error: {abs(ideal_result - noisy_result):.4f}")
benchmarks = [
    benchmark_zne(circuit),
    benchmark_pec(circuit),
    benchmark_cdr(circuit, training_circuits)
]

print(f"\n{'Method':<6} {'Time(s)':<8} {'Circuits':<9} {'Result':<8} {'Error':<8}")
print("-" * 50)
for b in benchmarks:
    if b['result'] is not None:
        error = abs(ideal_result - b['result'])
        print(f"{b['technique']:<6} {b['time']:.3f}    {b['circuits']:<9} {b['result']:.4f}   {error:.4f}")
    else:
        print(f"{b['technique']:<6} {b['time']:.3f}    {b['circuits']:<9} Failed   Failed")

print(f"\nBaseline error: {abs(ideal_result - noisy_result):.4f}")
print("ZNE: 3x overhead, reliable. PEC: High overhead, complex. CDR: Needs training data.")
```

### Example: Robust Circuit Execution
```python
import cirq
import time
from mitiq import zne
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.inference import LinearFactory
from mitiq.benchmarks import generate_rb_circuits
from cirq import DensityMatrixSimulator, depolarize
from mitiq.interface import convert_to_mitiq

def robust_execute(circuit, noise_level=0.01, max_retries=3):
    for attempt in range(max_retries):
        try:
            mitiq_circuit, _ = convert_to_mitiq(circuit)
            noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
            
            rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
            result = rho[0, 0].real
            
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed. Last error: {str(e)}")
                raise e
            print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(0.1)  # Brief pause before retry

def execute_with_zne_robust(circuit, scale_factors=[1.0, 2.0, 3.0], max_retries=3):
    try:
        print("Constructing ZNE circuits...")
        
        zne_circuits = zne.construct_circuits(
            circuit,
            scale_factors=scale_factors,
            scale_method=fold_gates_at_random  
        )
        
        print(f"Generated {len(zne_circuits)} circuits with scale factors {scale_factors}")
        
        results = []
        for i, circ in enumerate(zne_circuits):
            print(f"Executing circuit {i+1}/{len(zne_circuits)} (scale factor {scale_factors[i]})...")
            result = robust_execute(circ, max_retries=max_retries)
            results.append(result)
            print(f"  Result: {result:.6f}")
        
        factory = LinearFactory(scale_factors=scale_factors)
        mitigated_result = zne.combine_results(
            scale_factors,  # scale_factors first
            results,        # results second
            factory.extrapolate  # extrapolation method
        )
        
        return mitigated_result, results
        
    except Exception as e:
        print(f"Error during ZNE execution: {str(e)}")
        return None, []

def main():
    try:
        
        circuit = generate_rb_circuits(n_qubits=2, num_cliffords=30, return_type="cirq")[0]
        print(f"Generated circuit with {len(circuit)} operations")
        
        ideal_result = robust_execute(circuit, noise_level=0.0)
        noisy_result = robust_execute(circuit, noise_level=0.01)
        
        print(f"Ideal result: {ideal_result:.6f}")
        print(f"Noisy result: {noisy_result:.6f}")
        print(f"Error without mitigation: {abs(ideal_result - noisy_result):.6f}")
            mitigated_result, individual_results = execute_with_zne_robust(
            circuit, 
            scale_factors=[1.0, 2.0, 3.0],
            max_retries=3
        )
        
        if mitigated_result is not None:
            print(f"\nZNE Results:")
            print(f"Individual results: {[f'{r:.6f}' for r in individual_results]}")
            print(f"Mitigated result: {mitigated_result:.6f}")
            print(f"Error with ZNE: {abs(ideal_result - mitigated_result):.6f}")
            
            # Performance comparison
            error_reduction = abs(ideal_result - noisy_result) - abs(ideal_result - mitigated_result)
            improvement = (error_reduction / abs(ideal_result - noisy_result)) * 100
            print(f"Error reduction: {error_reduction:.6f} ({improvement:.1f}% improvement)")
        else:
            print("ZNE execution failed completely.")
            
        try:
            one_step_result = zne.execute_with_zne(
                circuit,
                lambda c: robust_execute(c, max_retries=1),  # Wrapper for compatibility
                factory=LinearFactory(scale_factors=[1.0, 2.0, 3.0]),
                scale_noise=fold_gates_at_random
            )
            print(f"One-step ZNE result: {one_step_result:.6f}")
            print(f"Error with one-step ZNE: {abs(ideal_result - one_step_result):.6f}")
        except Exception as e:
            print(f"One-step ZNE failed: {str(e)}")
            
    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```
## 7. Hardware-Specific Cost Analysis

### IBM Quantum
```python
# !pip install qiskit qiskit-ibm-runtime --quiet
# uncomment  this for installing qiskit framework's runtime


from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit

service = QiskitRuntimeService()

backend = service.backend("ibmq_qasm_simulator")  # Or 'ibm_kyoto', etc.

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

def analyze_ibm_costs(circuit):
    num_qubits = circuit.num_qubits
    depth = circuit.depth()

    # Dummy pricing model (IBM does not disclose real prices)
    base_cost = 0.0001
    qubit_cost = 0.00001
    depth_cost = 0.000001

    estimated_cost = base_cost + (num_qubits * qubit_cost) + (depth * depth_cost)

    return {
        "backend": backend.name,
        "num_qubits": num_qubits,
        "depth": depth,
        "estimated_cost_usd": round(estimated_cost, 8)
    }

print(analyze_ibm_costs(qc))

```
