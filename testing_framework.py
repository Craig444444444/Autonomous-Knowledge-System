import logging
import unittest
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import inspect
import traceback
import statistics
import hashlib

LOGGER = logging.getLogger("aks")

class AKSTestResult(unittest.TestResult):
    """Custom test result class with enhanced reporting capabilities."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_metrics = {}
        self.start_time = None
        self.current_test = None
        
    def startTest(self, test):
        self.start_time = time.time()
        self.current_test = test
        self.test_metrics[test] = {
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'duration': None,
            'memory_usage': [],
            'errors': []
        }
        super().startTest(test)
        
    def addSuccess(self, test):
        duration = time.time() - self.start_time
        self.test_metrics[test]['status'] = 'passed'
        self.test_metrics[test]['duration'] = duration
        super().addSuccess(test)
        
    def addError(self, test, err):
        duration = time.time() - self.start_time
        self._record_failure(test, err, 'error')
        self.test_metrics[test]['duration'] = duration
        super().addError(test, err)
        
    def addFailure(self, test, err):
        duration = time.time() - self.start_time
        self._record_failure(test, err, 'failure') 
        self.test_metrics[test]['duration'] = duration
        super().addFailure(test, err)
        
    def addSkip(self, test, reason):
        self.test_metrics[test]['status'] = 'skipped'
        self.test_metrics[test]['skip_reason'] = reason
        super().addSkip(test, reason)
        
    def _record_failure(self, test, err, failure_type):
        self.test_metrics[test]['status'] = failure_type
        exc_type, exc_value, exc_traceback = err
        self.test_metrics[test]['errors'].append({
            'type': str(exc_type),
            'message': str(exc_value),
            'traceback': traceback.format_tb(exc_traceback)
        })

class AKSTestRunner:
    """Enhanced test runner for AKS with performance tracking and reporting."""
    def __init__(self, output_dir: Path = None, verbosity: int = 1):
        self.output_dir = output_dir or Path("test_results")
        self.verbosity = verbosity
        self.results = {}
        
    def run(self, test_suite):
        """Execute test suite and generate comprehensive reports."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        result = AKSTestResult()
        start_time = time.time()
        
        LOGGER.info(f"Running {test_suite.countTestCases()} tests")
        test_suite.run(result)
        
        duration = time.time() - start_time
        self._generate_reports(result, duration)
        return result
        
    def _generate_reports(self, result, total_duration):
        """Generate various test reports and artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_data = {
            'timestamp': timestamp,
            'total_duration': total_duration,
            'tests': [],
            'summary': {
                'total': len(result.test_metrics),
                'passed': len([1 for t in result.test_metrics.values() if t['status'] == 'passed']),
                'failed': len([1 for t in result.test_metrics.values() if t['status'] == 'failure']),
                'errors': len([1 for t in result.test_metrics.values() if t['status'] == 'error']),
                'skipped': len([1 for t in result.test_metrics.values() if t['status'] == 'skipped']),
            },
            'performance': self._calculate_performance_metrics(result)
        }
        
        # Generate individual test reports
        for test, metrics in result.test_metrics.items():
            test_data = {
                'name': test.id(),
                'module': test.__module__,
                'description': test.shortDescription() or "",
                'status': metrics['status'],
                'duration': metrics['duration'],
                'start_time': metrics['start_time']
            }
            
            if metrics['status'] in ('failure', 'error'):
                test_data['errors'] = metrics['errors']
            elif metrics['status'] == 'skipped':
                test_data['skip_reason'] = metrics.get('skip_reason', '')
                
            report_data['tests'].append(test_data)
        
        # Save JSON report
        json_report = self.output_dir / f"test_report_{timestamp}.json"
        with open(json_report, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        # Save HTML report
        self._generate_html_report(report_data)
        
        # Save performance report
        self._generate_performance_report(report_data['performance'])
        
        LOGGER.info(f"Test results saved to {self.output_dir}")
        
    def _calculate_performance_metrics(self, result):
        """Calculate performance statistics from test results."""
        durations = [t['duration'] for t in result.test_metrics.values() if t['duration'] is not None]
        
        if not durations:
            return {}
            
        return {
            'mean': statistics.mean(durations),
            'median': statistics.median(durations),
            'min': min(durations),
            'max': max(durations),
            'total': sum(durations),
            'slowest_tests': sorted(
                [(test.id(), metrics['duration']) 
                 for test, metrics in result.test_metrics.items() 
                 if metrics['duration'] is not None],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        
    def _generate_html_report(self, report_data):
        """Generate an HTML version of the test report."""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AKS Test Report - {report_data['timestamp']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .test {{ margin-bottom: 10px; padding: 10px; border-left: 4px solid #ddd; }}
                .passed {{ border-color: #4CAF50; background-color: #e8f5e9; }}
                .failed {{ border-color: #f44336; background-color: #ffebee; }}
                .error {{ border-color: #FF9800; background-color: #fff3e0; }}
                .skipped {{ border-color: #2196F3; background-color: #e3f2fd; }}
                .performance {{ margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>AKS Test Report</h1>
            <p>Generated at {datetime.now().isoformat()}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {report_data['summary']['total']}</p>
                <p>Passed: {report_data['summary']['passed']}</p>
                <p>Failed: {report_data['summary']['failed']}</p>
                <p>Errors: {report_data['summary']['errors']}</p>
                <p>Skipped: {report_data['summary']['skipped']}</p>
                <p>Total Duration: {report_data['total_duration']:.2f} seconds</p>
            </div>
            
            <h2>Test Details</h2>
            {"".join(self._generate_test_html(test) for test in report_data['tests'])}
            
            <div class="performance">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Mean Duration</td><td>{report_data['performance']['mean']:.4f}s</td></tr>
                    <tr><td>Median Duration</td><td>{report_data['performance']['median']:.4f}s</td></tr>
                    <tr><td>Min Duration</td><td>{report_data['performance']['min']:.4f}s</td></tr>
                    <tr><td>Max Duration</td><td>{report_data['performance']['max']:.4f}s</td></tr>
                    <tr><td>Total Duration</td><td>{report_data['performance']['total']:.4f}s</td></tr>
                </table>
                
                <h3>Slowest Tests</h3>
                <table>
                    <tr><th>Test</th><th>Duration</th></tr>
                    {"".join(f"<tr><td>{test[0]}</td><td>{test[1]:.4f}s</td></tr>" 
                            for test in report_data['performance']['slowest_tests'])}
                </table>
            </div>
        </body>
        </html>
        """
        
        timestamp = report_data['timestamp']
        html_report = self.output_dir / f"test_report_{timestamp}.html"
        with open(html_report, 'w') as f:
            f.write(html_template)
            
    def _generate_test_html(self, test):
        """Generate HTML for an individual test result."""
        status_class = test['status']
        test_html = f"""
        <div class="test {status_class}">
            <h3>{test['name']}</h3>
            <p><strong>Module:</strong> {test['module']}</p>
            <p><strong>Status:</strong> <span class="{status_class}">{status_class.upper()}</span></p>
            <p><strong>Duration:</strong> {test['duration']:.4f}s</p>
        """
        
        if test['description']:
            test_html += f"<p><strong>Description:</strong> {test['description']}</p>"
            
        if test['status'] == 'skipped':
            test_html += f"<p><strong>Skip Reason:</strong> {test['skip_reason']}</p>"
            
        if test['status'] in ('failure', 'error') and 'errors' in test:
            test_html += "<h4>Errors:</h4><ul>"
            for error in test['errors']:
                test_html += f"""
                <li>
                    <p><strong>{error['type']}</strong>: {error['message']}</p>
                    <pre>{"".join(error['traceback'])}</pre>
                </li>
                """
            test_html += "</ul>"
            
        test_html += "</div>"
        return test_html
        
    def _generate_performance_report(self, performance_data):
        """Generate a dedicated performance report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        perf_report = self.output_dir / f"performance_report_{timestamp}.json"
        
        with open(perf_report, 'w') as f:
            json.dump(performance_data, f, indent=2)

# Explicitly define the alias needed for imports
TestingFramework = AKSTestRunner

class AKSTestLoader(unittest.TestLoader):
    """Custom test loader with enhanced discovery capabilities."""
    def __init__(self):
        super().__init__()
        self.test_method_prefix = 'test_'
        self.sortTestMethodsUsing = lambda a, b: a < b
        
    def discover(self, start_dir, pattern='test_*.py', top_level_dir=None):
        """Discover tests with additional logging and validation."""
        LOGGER.info(f"Discovering tests in {start_dir} with pattern {pattern}")
        return super().discover(start_dir, pattern, top_level_dir)
        
    def loadTestsFromModule(self, module):
        """Load tests from module with dependency checking."""
        LOGGER.debug(f"Loading tests from module {module.__name__}")
        return super().loadTestsFromModule(module)
        
    def loadTestsFromName(self, name, module=None):
        """Load tests from name with enhanced error handling."""
        try:
            return super().loadTestsFromName(name, module)
        except Exception as e:
            LOGGER.error(f"Failed to load test {name}: {e}")
            raise

class MockAKSComponent:
    """Base class for mocking AKS components during testing."""
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.responses = {}
        
    def __call__(self, *args, **kwargs):
        call_hash = self._hash_call(args, kwargs)
        self.calls.append({
            'args': args,
            'kwargs': kwargs,
            'timestamp': datetime.now().isoformat()
        })
        
        if call_hash in self.responses:
            return self.responses[call_hash]
            
        raise NotImplementedError(f"No mock response configured for {self.name} with args: {args}, kwargs: {kwargs}")
        
    def _hash_call(self, args, kwargs):
        """Create a hash of the call signature for response lookup."""
        call_str = json.dumps({
            'args': args,
            'kwargs': sorted(kwargs.items())
        }, sort_keys=True)
        return hashlib.md5(call_str.encode()).hexdigest()
        
    def add_response(self, args, kwargs, response):
        """Add a mock response for specific call signature."""
        call_hash = self._hash_call(args, kwargs)
        self.responses[call_hash] = response
        
    def assert_called_with(self, *args, **kwargs):
        """Assert that the mock was called with specific arguments."""
        expected_hash = self._hash_call(args, kwargs)
        for call in self.calls:
            if self._hash_call(call['args'], call['kwargs']) == expected_hash:
                return True
                
        raise AssertionError(f"Expected call not found in mock {self.name}")

class AKSTestCase(unittest.TestCase):
    """Base test case class for AKS with common utilities."""
    def setUp(self):
        self.start_time = time.time()
        self.test_id = self.id()
        LOGGER.info(f"Starting test: {self.test_id}")
        
    def tearDown(self):
        duration = time.time() - self.start_time
        LOGGER.info(f"Completed test: {self.test_id} in {duration:.4f}s")
        
    def assertKnowledgeEqual(self, first, second, msg=None):
        """Specialized assertion for knowledge items."""
        if isinstance(first, dict) and isinstance(second, dict):
            self.assertDictEqual(first, second, msg)
        elif hasattr(first, '__dict__') and hasattr(second, '__dict__'):
            self.assertDictEqual(first.__dict__, second.__dict__, msg)
        else:
            self.assertEqual(first, second, msg)
            
    def assertRepositoryState(self, repo_path, expected_files):
        """Verify repository contains expected files."""
        missing = []
        unexpected = []
        
        actual_files = {f.name for f in repo_path.glob('*') if f.is_file()}
        expected_set = set(expected_files)
        
        for expected in expected_set:
            if expected not in actual_files:
                missing.append(expected)
                
        for actual in actual_files:
            if actual not in expected_set:
                unexpected.append(actual)
                
        if missing or unexpected:
            msg = "Repository state mismatch\n"
            if missing:
                msg += f"Missing files: {missing}\n"
            if unexpected:
                msg += f"Unexpected files: {unexpected}"
            self.fail(msg)
            
    def mock_ai_response(self, prompt, response):
        """Mock an AI provider response."""
        mock_provider = MockAKSComponent("AIProvider")
        mock_provider.add_response((prompt,), {}, response)
        return mock_provider

def run_all_tests(test_dir: str = "tests", report_dir: str = "test_reports"):
    """Convenience function to discover and run all tests."""
    loader = AKSTestLoader()
    suite = loader.discover(test_dir)
    
    runner = AKSTestRunner(Path(report_dir))
    return runner.run(suite)

def generate_test_coverage(module_paths: List[str], output_file: Path):
    """Generate basic test coverage report."""
    coverage_data = {}
    
    for module_path in module_paths:
        module_path = Path(module_path)
        if not module_path.exists():
            continue
            
        coverage_data[module_path.name] = {
            'tested': False,
            'test_files': []
        }
        
        # Simple check for corresponding test file
        test_file = module_path.parent / f"test_{module_path.name}"
        if test_file.exists():
            coverage_data[module_path.name]['tested'] = True
            coverage_data[module_path.name]['test_files'].append(str(test_file))
            
    with open(output_file, 'w') as f:
        json.dump(coverage_data, f, indent=2)
        
    return coverage_data

class PerformanceBenchmark:
    """Utility class for performance benchmarking."""
    def __init__(self):
        self.measurements = {}
        
    def measure(self, name: str, func: Callable, *args, **kwargs):
        """Measure execution time of a function."""
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        if name not in self.measurements:
            self.measurements[name] = []
            
        self.measurements[name].append(duration)
        return result
        
    def get_stats(self):
        """Get statistics for all measurements."""
        stats = {}
        for name, times in self.measurements.items():
            if times:
                stats[name] = {
                    'count': len(times),
                    'mean': statistics.mean(times),
                    'median': statistics.median(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times)
                }
        return stats
