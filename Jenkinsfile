pipeline {
    agent any

    environment {
        PYTHON_EXE = "python"
        ALLURE_RESULTS_DIR = "allure-results"
        LINUX_ALLURE_RESULTS_DIR = "linux-allure-results"
        ALLURE_REPORT_DIR = "allure-report-latest"
        // Updated job name from 'robotics_tdd' to 'gpu_benchmark_suite'
        ALLURE_HISTORY_DIR = "C:\\ProgramData\\Jenkins\\.jenkins\\jobs\\gpu_benchmark_suite\\allure-history"
        PATH = "${env.PATH};${env.USERPROFILE}\\AppData\\Roaming\\npm"
        DOCKER_IMAGE = "python:3.10-slim" // Base image for dependency install
        BENCHMARK_TEST_SUITE = "gpu" // Pytest marker for the GPU tests
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout Source Code') {
            steps {
                // Updated URL to the GPU Benchmark repository (assumed)
                git url: 'https://github.com/luckyjoy/gpu_benchmark.git', branch: 'main'
            }
        }

        stage('Install Dependencies (Windows)') {
            steps {
                echo 'Installing required packages on Windows...'
                bat """
                    "%PYTHON_EXE%" -m pip install --upgrade pip
                    if exist requirements.txt "%PYTHON_EXE%" -m pip install -r requirements.txt
                    "%PYTHON_EXE%" -m pip install pytest allure-pytest
                    npm install -g allure-commandline --force
                    where allure >nul 2>nul || (echo Allure CLI not found on PATH. Ensure npm global bin is on PATH & exit /b 1)
                """
            }
        }

        stage('Load Allure History') {
            steps {
                echo "Restoring Allure history..."
                bat """
                    if exist "%ALLURE_RESULTS_DIR%" rd /s /q "%ALLURE_RESULTS_DIR%"
                    mkdir "%ALLURE_RESULTS_DIR%"
                    if exist "%ALLURE_HISTORY_DIR%" xcopy /E /I /Y "%ALLURE_HISTORY_DIR%" "%ALLURE_RESULTS_DIR%\\history" >nul
                """
            }
        }

        stage('Run Pytest (Windows)') {
            steps {
                echo "Running GPU Benchmark tests with Allure on Windows..."
                catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
                    // Changed marker from 'navigation' to '%BENCHMARK_TEST_SUITE%' ('gpu')
                    bat "\"%PYTHON_EXE%\" -m pytest -m %BENCHMARK_TEST_SUITE% --alluredir=\"%ALLURE_RESULTS_DIR%\" --capture=tee-sys"
                }
                echo "Windows tests execution complete."
            }
        }

		stage('Run Linux Tests in Docker (PowerShell)') {
			steps {
				catchError(buildResult: 'UNSTABLE', stageResult: 'UNSTABLE') {
					powershell '''
					$ErrorActionPreference = "Stop"

					Write-Host "========================================================="
					Write-Host "Checking Docker service and version..."
					docker version | Out-String | Write-Host
					Write-Host "========================================================="

					if (Test-Path "$env:WORKSPACE\\linux-allure-results") { Remove-Item -Recurse -Force "$env:WORKSPACE\\linux-allure-results" }
					New-Item -ItemType Directory -Force -Path "$env:WORKSPACE\\linux-allure-results" | Out-Null

					Write-Host "========================================================="
					Write-Host "Running GPU Benchmark Tests in Docker (Linux simulation)..."
					Write-Host "Docker Image: $env:DOCKER_IMAGE (Note: This assumes a GPU-enabled image or driver is pre-installed)"
					Write-Host "========================================================="

					docker pull $env:DOCKER_IMAGE

                    # Added --gpus all to enable passing GPU resources to the container.
                    # Changed the test marker to $env:BENCHMARK_TEST_SUITE.
					docker run --rm --gpus all -v "$env:WORKSPACE:/tests" -w /tests $env:DOCKER_IMAGE bash -lc \
						"pip install -q pytest allure-pytest && pip install -r requirements.txt && pytest -m $env:BENCHMARK_TEST_SUITE --alluredir=/tests/linux-allure-results"
					'''
				}
			}
		}

        stage('Add Allure Metadata') {
            steps {
                script {
                    echo 'Adding Allure metadata (categories, executor, environment)...'

                    // Updated Categories for GPU Benchmark Failures
                    def categoriesJson = """[
                        {
                            "name": "GPU/Driver Initialization Failure",
                            "messageRegex": ".*(CUDA|ROCm|DirectML|GPU initialization|No GPU found|Driver|Out of Memory).*",
                            "description": "Failures indicating a problem with the GPU or deep learning framework setup.",
                            "matchedStatuses": ["failed", "broken"]
                        },
                        {
                            "name": "Performance Regression / Assertion Failure",
                            "messageRegex": ".*(AssertionError|throughput|less than|greater than|slow|not approximately).*",
                            "description": "Failures related to performance thresholds or core logic asserts.",
                            "matchedStatuses": ["failed"]
                        },
                        {
                            "name": "Environment / Stability Issue",
                            "messageRegex": ".*(Timeout|ConnectionError|simulated_robot initialization failed|Image Pull Failed).*",
                            "traceRegex": ".*(PyBullet|Gazebo|mocked gripper|kubectl).*",
                            "description": "Failures related to the CI environment or Docker/Kubernetes connection errors.",
                            "matchedStatuses": ["broken"]
                        }
                    ]""".stripIndent()
                    writeFile file: "${ALLURE_RESULTS_DIR}/categories.json", text: categoriesJson

                    // Updated Executor for GPU Benchmark
                    def executorJson = """{
                        "name": "GPU Benchmark Runner",
                        "type": "CI_Pipeline",
                        "url": "${env.JENKINS_URL}",
                        "buildOrder": "${env.BUILD_ID}",
                        "buildName": "GPU Benchmark #${env.BUILD_ID}",
                        "buildUrl": "${env.BUILD_URL}",
                        "reportUrl": "${env.BUILD_URL}GPU-Benchmark-Allure-Report-Build-${env.BUILD_NUMBER}/index.html",
                        "data": {
                            "Validation Engineer": "TBD",
                            "Product Model": "DL-Benchmark-Suite",
                            "Test Framework": "pytest"
                        }
                    }""".stripIndent()
                    writeFile file: "${ALLURE_RESULTS_DIR}/executor.json", text: executorJson

                    // Updated Environment Properties for GPU Benchmark
                    def envProps = """Project=GPU Benchmark Simulation Framework
						Author=Bang Thien Nguyen
						Benchmark Suite=PyTorch/Tensorflow
						Target Hardware=Heterogeneous (NVIDIA, Intel, AMD, CPU)
						Operating System=Windows 11 / Linux (Docker)
						Docker_Image_Name=luckyjoy/gpu-benchmark-report:latest (Deployment Image)
						Docker_Build_Context=Repository Root (.)
						Docker_Runtime_Environment=Run Linux Tests in Docker (PowerShell)
						Docker_Key_Role=Provides isolated environment for Pytest and Allure data generation.
						Python Version=3.10.12
						Framework Version=1.0.0
						Test Type=Benchmark
						HTML Reporter=Allure Test Report 2.35.1
						Build Number=${env.BUILD_NUMBER}
						""".stripIndent()
                    writeFile file: "${ALLURE_RESULTS_DIR}/environment.properties", text: envProps
                }
            }
        }

        stage('Generate Allure Report') {
            steps {
                echo "Generating merged Allure HTML report..."
                bat """
                    if exist "%ALLURE_REPORT_DIR%" rd /s /q "%ALLURE_REPORT_DIR%"
                    mkdir "%ALLURE_REPORT_DIR%"
                    allure generate "%ALLURE_RESULTS_DIR%" "%LINUX_ALLURE_RESULTS_DIR%" -o "%ALLURE_REPORT_DIR%" --clean
                """
                // Persist history for next build
                bat """
                    if exist "%ALLURE_REPORT_DIR%\\history" (
                        if not exist "%ALLURE_HISTORY_DIR%" mkdir "%ALLURE_HISTORY_DIR%"
                        xcopy /E /I /Y "%ALLURE_REPORT_DIR%\\history" "%ALLURE_HISTORY_DIR%" >nul
                    )
                """
            }
        }

        stage('Archive & Publish Allure Report') {
            steps {
                script {
                    echo 'Archiving and publishing Allure report...'
                    archiveArtifacts artifacts: "${ALLURE_REPORT_DIR}/**/*", allowEmptyArchive: true
                    publishHTML(target: [
                        // Updated report name
                        reportName: "GPU-Benchmark-Allure-Report-Build-${env.BUILD_NUMBER}-CrossPlatform",
                        reportDir: "${ALLURE_REPORT_DIR}",
                        reportFiles: "index.html",
                        keepAll: true,
                        alwaysLinkToLastBuild: true
                    ])
                }
            }
        }
    }

    post {
        always {
            echo "Report (if archived): ${env.BUILD_URL}artifact/${ALLURE_REPORT_DIR}/index.html"
            cleanWs()
            echo "Pipeline finished."
        }
    }
}