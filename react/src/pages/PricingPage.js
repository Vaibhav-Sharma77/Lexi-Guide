import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Check, X, Star, Zap, Shield, Users, Crown } from 'lucide-react';

const PricingPage = () => {
  const [selectedCategory, setSelectedCategory] = React.useState('business');
  
  const categories = [
    { id: 'student', name: 'Student', icon: '🎓' },
    { id: 'individual', name: 'Individual', icon: '👤' },
    { id: 'business', name: 'Business', icon: '🏢' },
    { id: 'legal', name: 'Legal Firms', icon: '⚖️' },
    { id: 'enterprise', name: 'Enterprise', icon: '🏛️' }
  ];

  const allPlans = {
    student: [
      {
        name: 'Student Free',
        price: '₹0',
        period: 'forever',
        description: 'Perfect for law students and academic research',
        features: [
          '10 contract analyses per month',
          'Basic risk assessment',
          'Educational content access',
          'Study guides and tutorials',
          'Email support',
          'Academic templates'
        ],
        notIncluded: [
          'Advanced risk detection',
          'Commercial use',
          'Priority support'
        ],
        buttonText: 'Verify Student Status',
        popular: false,
        color: 'blue',
        badge: '🎓 Student Verified'
      },
      {
        name: 'Student Pro',
        price: '₹199',
        period: 'per month',
        description: 'Advanced features for serious law students',
        features: [
          '50 contract analyses per month',
          'Advanced AI risk detection',
          'Case study database access',
          'Legal research tools',
          'Priority student support',
          'Internship contract templates',
          'Career guidance resources'
        ],
        notIncluded: [
          'Commercial licensing',
          'API access'
        ],
        buttonText: 'Start Student Trial',
        popular: true,
        color: 'blue'
      }
    ],
    individual: [
      {
        name: 'Personal',
        price: '₹299',
        period: 'per month',
        description: 'For individuals and freelancers',
        features: [
          '25 contract analyses per month',
          'Personal contract templates',
          'Basic risk assessment',
          'Email support',
          'Standard processing',
          'Personal data protection'
        ],
        notIncluded: [
          'Team features',
          'API access',
          'Priority support'
        ],
        buttonText: 'Get Started',
        popular: false,
        color: 'green'
      },
      {
        name: 'Freelancer Pro',
        price: '₹599',
        period: 'per month',
        description: 'Perfect for professional freelancers',
        features: [
          '100 contract analyses per month',
          'Advanced risk detection',
          'Client contract templates',
          'Invoice and agreement tools',
          'Priority support',
          'Freelancer protection insights',
          'Payment terms analysis'
        ],
        notIncluded: [
          'Team collaboration',
          'API access'
        ],
        buttonText: 'Start Free Trial',
        popular: true,
        color: 'green'
      }
    ],
    business: [
      {
        name: 'Startup',
        price: '₹999',
        period: 'per month',
        description: 'Ideal for startups and small businesses',
        features: [
          '50 contract analyses per month',
          'Advanced AI risk detection',
          'Business contract templates',
          'Team collaboration (up to 5 users)',
          'Priority email support',
          'Vendor agreement analysis',
          'Compliance checking'
        ],
        notIncluded: [
          'Custom integrations',
          'Dedicated support'
        ],
        buttonText: 'Start Free Trial',
        popular: false,
        color: 'primary'
      },
      {
        name: 'Professional',
        price: '₹2,499',
        period: 'per month',
        description: 'For growing businesses and teams',
        features: [
          '200 contract analyses per month',
          'Advanced AI risk detection',
          'Team collaboration (up to 25 users)',
          'Custom contract templates',
          'Priority support',
          'Advanced analytics',
          'Contract lifecycle management',
          'Audit trail and compliance'
        ],
        notIncluded: [
          'White-label solution',
          'API access'
        ],
        buttonText: 'Start Free Trial',
        popular: true,
        color: 'primary'
      }
    ],
    legal: [
      {
        name: 'Law Firm Basic',
        price: '₹4,999',
        period: 'per month',
        description: 'For small to medium law firms',
        features: [
          '500 contract analyses per month',
          'Legal-specific AI models',
          'Client portal access',
          'Precedent database',
          'Legal research integration',
          'Case management tools',
          'Billable hours tracking',
          'Client confidentiality features'
        ],
        notIncluded: [
          'Custom AI training',
          'White-label solution'
        ],
        buttonText: 'Contact Sales',
        popular: false,
        color: 'purple'
      },
      {
        name: 'Law Firm Pro',
        price: '₹9,999',
        period: 'per month',
        description: 'For established law firms with high volume',
        features: [
          'Unlimited contract analyses',
          'Custom AI model training',
          'Advanced legal research tools',
          'Multi-jurisdiction support',
          'Client white-label portal',
          'Advanced reporting',
          'Dedicated legal support',
          'Custom integration support'
        ],
        notIncluded: [],
        buttonText: 'Contact Sales',
        popular: true,
        color: 'purple'
      }
    ],
    enterprise: [
      {
        name: 'Enterprise Basic',
        price: '₹19,999',
        period: 'per month',
        description: 'For large corporations and institutions',
        features: [
          'Unlimited contract analyses',
          'Custom risk profiles',
          'Enterprise-grade security',
          'API access and integrations',
          'Dedicated account manager',
          'Custom workflows',
          'Advanced analytics dashboard',
          'Multi-department access'
        ],
        notIncluded: [
          'On-premise deployment',
          'Custom AI development'
        ],
        buttonText: 'Contact Sales',
        popular: false,
        color: 'secondary'
      },
      {
        name: 'Enterprise Premium',
        price: 'Custom',
        period: 'pricing',
        description: 'Fully customizable enterprise solution',
        features: [
          'Everything in Enterprise Basic',
          'On-premise deployment option',
          'Custom AI model development',
          'White-label solution',
          'Global multi-language support',
          '24/7 dedicated support team',
          'Custom compliance frameworks',
          'Advanced security features'
        ],
        notIncluded: [],
        buttonText: 'Contact Sales',
        popular: true,
        color: 'secondary'
      }
    ]
  };

  const plans = allPlans[selectedCategory];

  const faqs = [
    {
      question: 'How do I qualify for student pricing?',
      answer: 'Students can access special pricing by verifying their status with a valid .edu email address or student ID. We support students from law schools, business schools, and other relevant academic institutions.'
    },
    {
      question: 'What\'s the difference between Individual and Business plans?',
      answer: 'Individual plans are designed for personal use and freelancers with basic collaboration needs. Business plans include team features, advanced analytics, and enterprise-grade security for organizations.'
    },
    {
      question: 'Do Legal Firms get specialized features?',
      answer: 'Yes! Legal firm plans include precedent databases, legal research integration, client portal access, billable hours tracking, and specialized AI models trained on legal documents.'
    },
    {
      question: 'Can I switch between different category plans?',
      answer: 'Yes, you can switch between plans and categories anytime. However, some features are category-specific (like student verification or legal firm tools) and may require re-verification.'
    },
    {
      question: 'Is there really a free plan for students?',
      answer: 'Yes! Verified students get 10 contract analyses per month for free, plus access to educational content, study guides, and academic templates.'
    },
    {
      question: 'Do you offer discounts for nonprofits and educational institutions?',
      answer: 'Yes! We offer 50% discounts for registered nonprofits and educational institutions. Students get special pricing, and we have custom packages for academic research.'
    },
    {
      question: 'What enterprise features are available?',
      answer: 'Enterprise plans include unlimited analyses, custom AI training, on-premise deployment options, white-label solutions, dedicated support teams, and custom compliance frameworks.'
    },
    {
      question: 'Can I upgrade from one category to another?',
      answer: 'Absolutely! You can upgrade from student to professional, individual to business, etc. Your data and settings will be preserved, and you\'ll gain access to new category-specific features.'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <section className="gradient-bg text-white py-20">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="text-4xl lg:text-5xl font-bold mb-6">
              Pricing for Everyone
            </h1>
            <p className="text-xl text-gray-100 max-w-2xl mx-auto mb-8">
              From students to enterprises - find the perfect plan for your contract analysis needs.
            </p>
            <div className="flex flex-wrap justify-center gap-4 mb-8">
              <div className="inline-flex items-center bg-white bg-opacity-20 rounded-full px-4 py-2">
                <span className="text-sm">🎓 Students</span>
              </div>
              <div className="inline-flex items-center bg-white bg-opacity-20 rounded-full px-4 py-2">
                <span className="text-sm">👤 Individuals</span>
              </div>
              <div className="inline-flex items-center bg-white bg-opacity-20 rounded-full px-4 py-2">
                <span className="text-sm">🏢 Businesses</span>
              </div>
              <div className="inline-flex items-center bg-white bg-opacity-20 rounded-full px-4 py-2">
                <span className="text-sm">⚖️ Legal Firms</span>
              </div>
              <div className="inline-flex items-center bg-white bg-opacity-20 rounded-full px-4 py-2">
                <span className="text-sm">🏛️ Enterprise</span>
              </div>
            </div>
            <div className="inline-flex items-center bg-white bg-opacity-20 rounded-full px-6 py-3">
              <Star className="w-5 h-5 text-yellow-300 mr-2" />
              <span className="text-sm">Trusted by 2,500+ users across all categories</span>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Category Selector */}
      <section className="py-12">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Choose Your Category
            </h2>
            <p className="text-gray-600 mb-8">
              Select the plan category that best describes you for tailored pricing and features
            </p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
            {categories.map((category) => (
              <motion.button
                key={category.id}
                onClick={() => setSelectedCategory(category.id)}
                className={`p-4 rounded-xl border-2 transition-all duration-200 ${
                  selectedCategory === category.id
                    ? 'border-primary-500 bg-primary-50 text-primary-700'
                    : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="text-2xl mb-2">{category.icon}</div>
                <div className="text-sm font-medium">{category.name}</div>
              </motion.button>
            ))}
          </div>
          
          {selectedCategory === 'student' && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
              <p className="text-blue-800 text-sm">
                🎓 <strong>Student Discount:</strong> Verify your student status with a valid .edu email 
                or student ID to access special pricing and educational features.
              </p>
            </div>
          )}
          
          {selectedCategory === 'legal' && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 text-center">
              <p className="text-purple-800 text-sm">
                ⚖️ <strong>Legal Professional:</strong> Specialized features for legal practitioners 
                including precedent databases, legal research tools, and compliance features.
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Pricing Cards */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {plans.map((plan, index) => (
              <motion.div
                key={plan.name}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className={`relative bg-white rounded-2xl shadow-lg overflow-hidden ${
                  plan.popular ? 'ring-2 ring-primary-500 transform scale-105' : ''
                }`}
              >
                {plan.popular && (
                  <div className="absolute top-0 left-0 right-0 bg-primary-500 text-white text-center py-2">
                    <span className="text-sm font-medium flex items-center justify-center">
                      <Crown className="w-4 h-4 mr-1" />
                      Most Popular
                    </span>
                  </div>
                )}
                
                <div className={`p-8 ${plan.popular ? 'pt-16' : ''}`}>
                  {/* Plan Header */}
                  <div className="text-center mb-8">
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">{plan.name}</h3>
                    <p className="text-gray-600 mb-4">{plan.description}</p>
                    <div className="flex items-baseline justify-center">
                      <span className="text-4xl font-bold text-gray-900">{plan.price}</span>
                      <span className="text-gray-600 ml-2">/{plan.period}</span>
                    </div>
                  </div>

                  {/* Features */}
                  <div className="space-y-4 mb-8">
                    {plan.features.map((feature, featureIndex) => (
                      <div key={featureIndex} className="flex items-center">
                        <Check className="w-5 h-5 text-green-500 mr-3 flex-shrink-0" />
                        <span className="text-gray-700">{feature}</span>
                      </div>
                    ))}
                    {plan.notIncluded.map((feature, featureIndex) => (
                      <div key={featureIndex} className="flex items-center opacity-50">
                        <X className="w-5 h-5 text-gray-400 mr-3 flex-shrink-0" />
                        <span className="text-gray-500">{feature}</span>
                      </div>
                    ))}
                  </div>

                  {/* CTA Button */}
                  <Link
                    to={plan.name === 'Enterprise' ? '/contact' : '/analyze'}
                    className={`block w-full text-center py-4 px-6 rounded-lg font-semibold transition-all duration-200 ${
                      plan.popular
                        ? 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg hover:shadow-xl'
                        : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
                    }`}
                  >
                    {plan.buttonText}
                  </Link>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Comparison */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-6">
              Why Choose Lexi-Guide?
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Compare our features with traditional legal services
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                icon: Zap,
                title: 'Instant Analysis',
                traditional: '2-5 days',
                lexiGuide: '30 seconds',
                color: 'text-yellow-600'
              },
              {
                icon: Shield,
                title: 'Cost per Analysis',
                traditional: '₹5,000 - ₹25,000',
                lexiGuide: '₹20 - ₹100',
                color: 'text-green-600'
              },
              {
                icon: Users,
                title: 'Availability',
                traditional: 'Business hours only',
                lexiGuide: '24/7 instant access',
                color: 'text-blue-600'
              }
            ].map((comparison, index) => (
              <motion.div
                key={comparison.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-gray-50 rounded-xl p-8 text-center"
              >
                <comparison.icon className={`w-12 h-12 ${comparison.color} mx-auto mb-4`} />
                <h3 className="text-xl font-bold text-gray-900 mb-4">{comparison.title}</h3>
                <div className="space-y-3">
                  <div>
                    <div className="text-sm text-gray-600">Traditional Legal</div>
                    <div className="text-lg font-medium text-gray-800">{comparison.traditional}</div>
                  </div>
                  <div className="border-t pt-3">
                    <div className="text-sm text-primary-600">Lexi-Guide</div>
                    <div className="text-lg font-bold text-primary-600">{comparison.lexiGuide}</div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-6">
              Frequently Asked Questions
            </h2>
            <p className="text-lg text-gray-600">
              Have questions? We've got answers.
            </p>
          </motion.div>

          <div className="space-y-6">
            {faqs.map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white rounded-lg p-6 shadow-sm"
              >
                <h3 className="text-lg font-semibold text-gray-900 mb-3">{faq.question}</h3>
                <p className="text-gray-600">{faq.answer}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 gradient-bg text-white">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl lg:text-4xl font-bold mb-6">
              Ready to Get Started?
            </h2>
            <p className="text-xl text-gray-100 mb-8">
              Join thousands of smart businesses making informed legal decisions
            </p>
            <Link
              to="/analyze"
              className="inline-flex items-center bg-white text-primary-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-gray-100 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Start Your Free Analysis
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default PricingPage;