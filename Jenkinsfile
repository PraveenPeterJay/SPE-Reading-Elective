pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Run Ansible Automation') {
            steps {
                sh 'ansible-playbook playbook.yml -vvv 2>&1 | grep -A5 "apt cache"'
            }
        }
    }

    post {
        always {
            // This ensures results are saved even if a test or build fails
            echo "Archiving research data..."
            archiveArtifacts artifacts: '*.json, *.txt', allowEmptyArchive: true
        }
    }
}