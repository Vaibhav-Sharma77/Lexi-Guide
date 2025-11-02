import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Copy, 
  Download, 
  Share2, 
  RefreshCw,
  TrendingUp,
  Eye,
  FileText,
  HelpCircle,
  Star,
  ThumbsUp,
  ThumbsDown
} from 'lucide-react';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const AnalysisResults = ({ 
  analysisData, 
  contractText, 
  userRole, 
  selectedClause, 
  onClauseSelect, 
  onReset 
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [showFullContract, setShowFullContract] = useState(false);

  const safetyScore = analysisData.legal_safety_index.score;

  const getScoreColor = (score) => {
    if (score >= 70) return 'text-green-600';
    if (score >= 40) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreRingColor = (score) => {
    if (score >= 70) return '#22c55e';
    if (score >= 40) return '#f59e0b';
    return '#ef4444';
  };

  const getScoreBgColor = (score) => {
    if (score >= 70) return 'bg-green-50 border-green-200';
    if (score >= 40) return 'bg-yellow-50 border-yellow-200';
    return 'bg-red-50 border-red-200';
  };

  const getRiskIcon = (level) => {
    switch (level.toLowerCase()) {
      case 'high': return <AlertTriangle className="w-5 h-5 text-red-600" />;
      case 'medium': return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
      case 'low': return <CheckCircle className="w-5 h-5 text-green-600" />;
      default: return <CheckCircle className="w-5 h-5 text-gray-600" />;
    }
  };

  const getRiskClasses = (level) => {
    switch (level.toLowerCase()) {
      case 'high': return 'risk-high';
      case 'medium': return 'risk-medium';
      case 'low': return 'risk-low';
      default: return 'risk-none';
    }
  };

  const copyToClipboard = async (text, message = 'Copied to clipboard!') => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success(message);
    } catch (err) {
      toast.error('Failed to copy text');
    }
  };

  const downloadReport = async () => {
    try {
      const element = document.getElementById('analysis-report');
      const canvas = await html2canvas(element);
      const imgData = canvas.toDataURL('image/png');
      
      const pdf = new jsPDF();
      const imgWidth = 210;
      const pageHeight = 295;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      let heightLeft = imgHeight;
      
      let position = 0;
      
      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;
      
      while (heightLeft >= 0) {
        position = heightLeft - imgHeight;
        pdf.addPage();
        pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;
      }
      
      pdf.save('contract-analysis-report.pdf');
      toast.success('Report downloaded successfully!');
    } catch (error) {
      toast.error('Failed to download report');
    }
  };

  const shareReport = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Contract Analysis Report',
          text: `Legal Safety Score: ${safetyScore}/100 - Contract Analysis by Lexi-Guide`,
          url: window.location.href
        });
      } catch (error) {
        copyToClipboard(window.location.href, 'Report link copied to clipboard!');
      }
    } else {
      copyToClipboard(window.location.href, 'Report link copied to clipboard!');
    }
  };

  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (safetyScore / 100) * circumference;

  const riskCounts = {
    High: analysisData.clauses.filter(c => c.risk_level === 'High').length,
    Medium: analysisData.clauses.filter(c => c.risk_level === 'Medium').length,
    Low: analysisData.clauses.filter(c => c.risk_level === 'Low').length,
    None: analysisData.clauses.filter(c => c.risk_level === 'None').length
  };

  return (
    <div id="analysis-report" className="space-y-8">
      {/* Header with Actions */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="bg-white rounded-xl shadow-lg p-6"
      >
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center space-y-4 lg:space-y-0">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Contract Analysis Complete
            </h2>
            <p className="text-gray-600">
              Your contract has been analyzed with personalized insights for {userRole}s
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => copyToClipboard(JSON.stringify(analysisData, null, 2), 'Analysis data copied!')}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              <Copy className="w-4 h-4" />
              <span>Copy Data</span>
            </button>
            <button
              onClick={downloadReport}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors"
            >
              <Download className="w-4 h-4" />
              <span>Download PDF</span>
            </button>
            <button
              onClick={shareReport}
              className="flex items-center space-x-2 px-4 py-2 bg-green-100 hover:bg-green-200 text-green-700 rounded-lg transition-colors"
            >
              <Share2 className="w-4 h-4" />
              <span>Share</span>
            </button>
            <button
              onClick={onReset}
              className="flex items-center space-x-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span>New Analysis</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* Safety Score Dashboard */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        className={`rounded-xl shadow-lg p-8 border-2 ${getScoreBgColor(safetyScore)}`}
      >
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-center">
          {/* Score Circle */}
          <div className="flex justify-center">
            <div className="relative w-40 h-40">
              <svg className="transform -rotate-90 w-40 h-40">
                <circle
                  cx="80"
                  cy="80"
                  r="45"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="none"
                  className="text-gray-200"
                />
                <circle
                  cx="80"
                  cy="80"
                  r="45"
                  stroke={getScoreRingColor(safetyScore)}
                  strokeWidth="8"
                  fill="none"
                  strokeLinecap="round"
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                  className="safety-score-ring"
                  style={{
                    strokeDashoffset: strokeDashoffset
                  }}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className={`text-4xl font-bold ${getScoreColor(safetyScore)}`}>
                    {safetyScore}
                  </div>
                  <div className="text-sm text-gray-600">Safety Score</div>
                </div>
              </div>
            </div>
          </div>

          {/* Score Information */}
          <div className="lg:col-span-2">
            <div className="space-y-4">
              <div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">
                  Legal Safety Assessment
                </h3>
                <p className="text-gray-700">
                  {analysisData.legal_safety_index.justification}
                </p>
              </div>

              {/* Risk Summary */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {Object.entries(riskCounts).map(([level, count]) => (
                  <div key={level} className="text-center">
                    <div className={`text-2xl font-bold ${
                      level === 'High' ? 'text-red-600' :
                      level === 'Medium' ? 'text-yellow-600' :
                      level === 'Low' ? 'text-green-600' :
                      'text-gray-600'
                    }`}>
                      {count}
                    </div>
                    <div className="text-sm text-gray-600">{level} Risk</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Tabs */}
      <div className="bg-white rounded-xl shadow-lg">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {[
              { id: 'overview', label: 'Overview', icon: TrendingUp },
              { id: 'clauses', label: 'Clause Analysis', icon: FileText },
              { id: 'contract', label: 'Full Contract', icon: Eye }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="space-y-6"
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* High Priority Issues */}
                <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-red-900 mb-4 flex items-center">
                    <AlertTriangle className="w-5 h-5 mr-2" />
                    High Priority Issues ({riskCounts.High})
                  </h3>
                  <div className="space-y-3">
                    {analysisData.clauses
                      .filter(clause => clause.risk_level === 'High')
                      .slice(0, 3)
                      .map((clause, index) => (
                        <div key={index} className="text-sm">
                          <button
                            onClick={() => {
                              onClauseSelect(clause);
                              setActiveTab('clauses');
                            }}
                            className="text-red-800 hover:text-red-900 font-medium text-left"
                          >
                            {clause.clause_title}
                          </button>
                        </div>
                      ))}
                    {riskCounts.High > 3 && (
                      <button
                        onClick={() => setActiveTab('clauses')}
                        className="text-red-600 hover:text-red-700 text-sm font-medium"
                      >
                        View {riskCounts.High - 3} more...
                      </button>
                    )}
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-blue-900 mb-4 flex items-center">
                    <Star className="w-5 h-5 mr-2" />
                    Key Recommendations
                  </h3>
                  <div className="space-y-3 text-sm text-blue-800">
                    {safetyScore < 40 && (
                      <div>• Consider significant revisions before signing</div>
                    )}
                    {riskCounts.High > 0 && (
                      <div>• Negotiate high-risk clauses immediately</div>
                    )}
                    <div>• Review payment terms carefully</div>
                    <div>• Ensure liability limitations are in place</div>
                    <div>• Seek legal counsel for final review</div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Clauses Tab */}
          {activeTab === 'clauses' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="space-y-6"
            >
              {analysisData.clauses.map((clause, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  className={`p-6 rounded-lg border clause-highlight ${getRiskClasses(clause.risk_level)} ${
                    selectedClause === clause ? 'ring-2 ring-primary-500' : ''
                  }`}
                  onClick={() => onClauseSelect(clause)}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      {getRiskIcon(clause.risk_level)}
                      <h3 className="text-lg font-bold text-gray-900">
                        {clause.clause_title}
                      </h3>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                      clause.risk_level === 'High' ? 'bg-red-100 text-red-800' :
                      clause.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                      clause.risk_level === 'Low' ? 'bg-green-100 text-green-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {clause.risk_level} Risk
                    </span>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium text-gray-700 mb-2">Original Clause:</h4>
                      <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded">
                        "{clause.original_text}"
                      </p>
                    </div>

                    <div>
                      <h4 className="font-medium text-gray-700 mb-2">What this means for you:</h4>
                      <p className="text-sm text-gray-700 bg-blue-50 p-3 rounded">
                        {clause.impact_first_explanation}
                      </p>
                    </div>

                    {clause.safer_suggestion !== "No suggestion needed. This clause is well-balanced." && (
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium text-gray-700">Suggested Improvement:</h4>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              copyToClipboard(clause.safer_suggestion, 'Suggestion copied!');
                            }}
                            className="text-gray-500 hover:text-gray-700"
                          >
                            <Copy className="w-4 h-4" />
                          </button>
                        </div>
                        <p className="text-sm text-gray-700 bg-green-50 p-3 rounded border border-green-200">
                          {clause.safer_suggestion}
                        </p>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}

          {/* Full Contract Tab */}
          {activeTab === 'contract' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
            >
              <div className="bg-gray-50 p-6 rounded-lg">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-bold text-gray-900">Full Contract Text</h3>
                  <button
                    onClick={() => copyToClipboard(contractText, 'Contract text copied!')}
                    className="flex items-center space-x-2 px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded text-sm transition-colors"
                  >
                    <Copy className="w-4 h-4" />
                    <span>Copy</span>
                  </button>
                </div>
                <div className="bg-white p-4 rounded border max-h-96 overflow-y-auto">
                  <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono">
                    {contractText}
                  </pre>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;