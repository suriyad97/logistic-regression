# Azure Bicep template for deploying ML training infrastructure
# Creates AML workspace, compute cluster, storage, and networking resources

param location string = resourceGroup().location
param workspaceName string
param storageAccountName string
param keyVaultName string
param appInsightsName string
param computeClusterName string = 'cpu-cluster'
param computeInstanceType string = 'Standard_D2s_v3'
param computeMinNodes int = 0
param computeMaxNodes int = 10

variable uniqueSuffix = uniqueString(resourceGroup().id)

// Storage Account
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: '${storageAccountName}${uniqueSuffix}'
  location: location
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    accessTier: 'Hot'
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

// Create blob container
resource blobContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/amldata'
  properties: {
    publicAccess: 'None'
  }
}

// Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: '${keyVaultName}${uniqueSuffix}'
  location: location
  properties: {
    enabledForDeployment: true
    enabledForTemplateDeployment: true
    enabledForDiskEncryption: false
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    accessPolicies: []
    enablePurgeProtection: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
  }
}

// Application Insights
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${appInsightsName}${uniqueSuffix}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    RetentionInDays: 90
  }
}

// Container Registry
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: 'acr${replace(uniqueSuffix, '-', '')}'
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
    anonymousPullEnabled: false
    dataEndpointEnabled: false
    encryption: {
      enabled: false
    }
    networkRuleBypassOptions: 'AzureServices'
    networkRuleSetDefaultAction: 'Allow'
  }
}

// Azure ML Workspace
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = {
  name: '${workspaceName}${uniqueSuffix}'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: workspaceName
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: containerRegistry.id
    hbiWorkspace: false
    imageBuildCompute: null
    publicNetworkAccess: 'Enabled'
  }
}

// Compute Cluster
resource computeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2023-04-01' = {
  parent: mlWorkspace
  name: computeClusterName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: computeInstanceType
      vmPriority: 'Dedicated'
      minNodeCount: computeMinNodes
      maxNodeCount: computeMaxNodes
      nodeIdleTimeBeforeScaleDown: 900
    }
  }
}

// Output important values
output workspaceId string = mlWorkspace.id
output workspaceName string = mlWorkspace.name
output storageAccountId string = storageAccount.id
output keyVaultId string = keyVault.id
output containerRegistryId string = containerRegistry.id
