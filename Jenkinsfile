pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/PraveenPeterJay/SPE-Reading-Elective.git'
            }
        }

        stage('Run Ansible Automation') {
            steps {
                // This calls the playbook we wrote above
                ansiblePlaybook playbook: 'pipeline.yml'
            }
        }

        stage('Archive Artifacts') {
            steps {
                // This saves your test results in Jenkins so you can download them
                archiveArtifacts artifacts: '*.json, *.txt', allowEmptyArchive: true
            }
        }
    }
}