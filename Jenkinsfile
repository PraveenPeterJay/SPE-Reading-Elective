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
                sh "ansible-playbook -i ansible/inventory.ini ansible/pipeline.yml"
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