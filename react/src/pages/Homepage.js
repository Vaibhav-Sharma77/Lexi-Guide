import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Shield, 
  Zap, 
  Users, 
  CheckCircle, 
  Star,
  ArrowRight,
  FileText,
  Clock,
  TrendingUp,
  Award
} from 'lucide-react';

const Homepage = () => {
  const features = [
    {
      icon: Shield,
      title: 'Legal Safety Index™',
      description: 'Get instant 0-100 safety scores for any contract with AI-powered risk assessment.',
      color: 'text-blue-600'
    },
    {
      icon: Zap,
      title: 'Instant Analysis',
      description: 'Upload contracts in seconds and receive comprehensive analysis in under 30 seconds.',
      color: 'text-yellow-600'
    },
    {
      icon: FileText,
      title: 'Multi-Format Support',
      description: 'Analyze PDF, DOC, DOCX, and text files with our advanced document processing.',
      color: 'text-green-600'
    },
    {
      icon: Users,
      title: 'MSME Focused',
      description: 'Specially designed for Indian small and medium enterprises with local legal context.',
      color: 'text-purple-600'
    }
  ];

  const testimonials = [
    {
      name: 'Ananya Singh',
      role: 'Law Student, NLSIU',
      image: '/api/placeholder/64/64',
      content: 'As a law student, Lexi-Guide helps me understand complex contracts and learn practical legal analysis. The educational features are amazing!',
      rating: 5,
      category: 'Student'
    },
    {
      name: 'Rajesh Kumar',
      role: 'Founder, TechStart Solutions',
      image: '/api/placeholder/64/64',
      content: 'Lexi-Guide saved us from a terrible contract that would have cost us lakhs. The AI spotted risks our lawyer missed!',
      rating: 5,
      category: 'Business'
    },
    {
      name: 'Priya Sharma',
      role: 'Freelance Designer',
      image: '/api/placeholder/64/64',
      content: 'As a freelancer, this tool gives me confidence in every contract negotiation. The individual plan is perfect for my needs!',
      rating: 5,
      category: 'Individual'
    },
    {
      name: 'Advocate Suresh Menon',
      role: 'Senior Partner, Menon & Associates',
      image: '/api/placeholder/64/64',
      content: 'The legal research integration and precedent database have significantly enhanced our firm\'s efficiency. Excellent for legal professionals.',
      rating: 5,
      category: 'Legal Firm'
    },
    {
      name: 'Arjun Patel',
      role: 'Director, Manufacturing Co.',
      image: '/api/placeholder/64/64',
      content: 'The enterprise features and custom workflows have streamlined our contract management process across multiple departments.',
      rating: 5,
      category: 'Enterprise'
    }
  ];

  const stats = [
    { label: 'Contracts Analyzed', value: '10,000+' },
    { label: 'Businesses Protected', value: '2,500+' },
    { label: 'Risk Issues Found', value: '50,000+' },
    { label: 'Money Saved', value: '₹5 Cr+' }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="gradient-bg text-white py-20 lg:py-32 relative overflow-hidden">
        <div className="absolute inset-0 bg-black bg-opacity-20"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-4xl lg:text-6xl font-bold mb-6">
              Your AI Legal Co-Pilot for
              <span className="block text-yellow-300">Smarter Contracts</span>
            </h1>
            <p className="text-xl lg:text-2xl mb-8 max-w-3xl mx-auto text-gray-100">
              For students, professionals, businesses, and legal firms - 
              analyze contracts in seconds, spot risks instantly, and negotiate with confidence.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/analyze"
                className="bg-white text-primary-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-gray-100 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 flex items-center justify-center"
              >
                Start Free Analysis
                <ArrowRight className="ml-2 w-5 h-5" />
              </Link>
              <Link
                to="/about"
                className="border-2 border-white text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-white hover:text-primary-600 transition-all duration-300 flex items-center justify-center"
              >
                Learn More
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-3xl lg:text-4xl font-bold text-primary-600 mb-2">
                  {stat.value}
                </div>
                <div className="text-gray-600">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-5xl font-bold text-gray-900 mb-6">
              Why Choose Lexi-Guide?
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Powered by advanced AI and designed specifically for Indian businesses, 
              we make complex legal documents simple to understand.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
              >
                <feature.icon className={`w-12 h-12 ${feature.color} mb-4`} />
                <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* User Categories Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-5xl font-bold text-gray-900 mb-6">
              Built for Everyone
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Whether you're a student, professional, or enterprise - we have the right solution for you
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
            {[
              {
                icon: '🎓',
                title: 'Students',
                description: 'Learn contract law with AI guidance, access educational resources, and analyze academic contracts.',
                features: ['Free tier available', 'Educational content', 'Study guides'],
                color: 'bg-blue-50 border-blue-200'
              },
              {
                icon: '👤',
                title: 'Individuals',
                description: 'Freelancers and professionals protecting their interests in personal and work contracts.',
                features: ['Personal contracts', 'Freelancer protection', 'Simple pricing'],
                color: 'bg-green-50 border-green-200'
              },
              {
                icon: '🏢',
                title: 'Businesses',
                description: 'SMEs and startups managing vendor agreements, partnerships, and commercial contracts.',
                features: ['Team collaboration', 'Business templates', 'Compliance tools'],
                color: 'bg-primary-50 border-primary-200'
              },
              {
                icon: '⚖️',
                title: 'Legal Firms',
                description: 'Law firms enhancing their practice with AI-powered contract analysis and client tools.',
                features: ['Legal research', 'Client portals', 'Precedent database'],
                color: 'bg-purple-50 border-purple-200'
              },
              {
                icon: '🏛️',
                title: 'Enterprise',
                description: 'Large organizations with complex contract workflows and enterprise-grade security needs.',
                features: ['Custom solutions', 'API access', 'Dedicated support'],
                color: 'bg-gray-50 border-gray-200'
              }
            ].map((category, index) => (
              <motion.div
                key={category.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className={`${category.color} border-2 rounded-xl p-6 text-center hover:shadow-lg transition-all duration-300`}
              >
                <div className="text-4xl mb-4">{category.icon}</div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">{category.title}</h3>
                <p className="text-gray-600 mb-4 text-sm">{category.description}</p>
                <ul className="space-y-1">
                  {category.features.map((feature, idx) => (
                    <li key={idx} className="text-xs text-gray-500 flex items-center justify-center">
                      <CheckCircle className="w-3 h-3 mr-1 text-green-500" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="text-center mt-12"
          >
            <Link
              to="/pricing"
              className="inline-flex items-center bg-primary-600 text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-primary-700 transition-all duration-300 shadow-lg hover:shadow-xl"
            >
              View All Pricing Plans
              <ArrowRight className="ml-2 w-5 h-5" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-5xl font-bold text-gray-900 mb-6">
              How It Works
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Get professional legal insights in just three simple steps
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: '1',
                title: 'Upload Contract',
                description: 'Drag and drop your contract or paste the text. We support PDF, DOC, DOCX, and plain text.',
                icon: FileText
              },
              {
                step: '2',
                title: 'AI Analysis',
                description: 'Our advanced AI analyzes every clause, identifies risks, and calculates a safety score.',
                icon: Zap
              },
              {
                step: '3',
                title: 'Get Insights',
                description: 'Receive detailed explanations, risk assessments, and safer alternative clauses.',
                icon: CheckCircle
              }
            ].map((item, index) => (
              <motion.div
                key={item.step}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                className="text-center"
              >
                <div className="w-16 h-16 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full flex items-center justify-center text-white text-2xl font-bold mx-auto mb-6">
                  {item.step}
                </div>
                <item.icon className="w-12 h-12 text-primary-600 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-gray-900 mb-3">{item.title}</h3>
                <p className="text-gray-600">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-5xl font-bold text-gray-900 mb-6">
              What Our Users Say
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Join thousands of businesses who trust Lexi-Guide for their legal decisions
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={testimonial.name}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white p-6 rounded-xl shadow-lg"
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} className="w-4 h-4 text-yellow-400 fill-current" />
                    ))}
                  </div>
                  <span className="bg-primary-100 text-primary-800 text-xs px-2 py-1 rounded-full font-medium">
                    {testimonial.category}
                  </span>
                </div>
                <p className="text-gray-700 mb-4 text-sm">"{testimonial.content}"</p>
                <div className="flex items-center">
                  <div className="w-10 h-10 bg-gray-300 rounded-full mr-3"></div>
                  <div>
                    <div className="font-semibold text-gray-900 text-sm">{testimonial.name}</div>
                    <div className="text-gray-600 text-xs">{testimonial.role}</div>
                  </div>
                </div>
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
            <h2 className="text-3xl lg:text-5xl font-bold mb-6">
              Ready to Analyze Your First Contract?
            </h2>
            <p className="text-xl mb-8 text-gray-100">
              Join thousands of smart businesses making informed legal decisions with AI
            </p>
            <Link
              to="/analyze"
              className="inline-flex items-center bg-white text-primary-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-gray-100 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Start Free Analysis Now
              <ArrowRight className="ml-2 w-5 h-5" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Homepage;