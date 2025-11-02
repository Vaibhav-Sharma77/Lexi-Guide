import React from 'react';
import { motion } from 'framer-motion';
import { Shield, Users, Target, Award, Heart, Zap } from 'lucide-react';

const AboutPage = () => {
  const team = [
    {
      name: 'Suraj Mishra',
      role: 'Project Lead & AI Developer',
      image: '/api/placeholder/150/150',
      bio: 'Lead developer and AI specialist, passionate about legal technology innovation'
    },
    {
      name: 'Disha',
      role: 'Frontend Developer',
      image: '/api/placeholder/150/150',
      bio: 'UI/UX specialist focused on creating intuitive user experiences'
    },
    {
      name: 'Vaibhav',
      role: 'Backend Developer',
      image: '/api/placeholder/150/150',
      bio: 'Backend architect ensuring robust and scalable API development'
    },
    {
      name: 'Kashish',
      role: 'Product Designer',
      image: '/api/placeholder/150/150',
      bio: 'Product design and user research expert'
    }
  ];

  const values = [
    {
      icon: Shield,
      title: 'Trust & Security',
      description: 'Your contracts and data are handled with enterprise-grade security and complete confidentiality.'
    },
    {
      icon: Users,
      title: 'MSME First',
      description: 'Built specifically for Indian small and medium enterprises, understanding their unique challenges.'
    },
    {
      icon: Target,
      title: 'Accuracy',
      description: 'Our AI is trained on thousands of Indian contracts, ensuring precise and relevant analysis.'
    },
    {
      icon: Heart,
      title: 'Empowerment',
      description: 'We believe everyone deserves access to quality legal insights, not just large corporations.'
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
              About Lexi-Guide
            </h1>
            <p className="text-xl text-gray-100 max-w-2xl mx-auto">
              We're on a mission to democratize legal insights and empower every Indian business 
              to make informed legal decisions with confidence.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-6">
                Our Mission
              </h2>
              <p className="text-lg text-gray-700 mb-6">
                In India's rapidly growing economy, MSMEs form the backbone of business. Yet, 
                complex legal contracts often create barriers to growth, leading to unfavorable 
                terms and missed opportunities.
              </p>
              <p className="text-lg text-gray-700 mb-6">
                Lexi-Guide bridges this gap by making AI-powered legal analysis accessible to 
                everyone. We transform legal anxiety into confident action, helping businesses 
                negotiate better terms and protect their interests.
              </p>
              <div className="flex items-center space-x-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-primary-600">10K+</div>
                  <div className="text-sm text-gray-600">Contracts Analyzed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-primary-600">2.5K+</div>
                  <div className="text-sm text-gray-600">Businesses Protected</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-primary-600">₹5Cr+</div>
                  <div className="text-sm text-gray-600">Money Saved</div>
                </div>
              </div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="bg-gradient-to-br from-primary-500 to-secondary-500 rounded-2xl p-8 text-white"
            >
              <Zap className="w-12 h-12 mb-6" />
              <h3 className="text-2xl font-bold mb-4">Why We Started</h3>
              <p className="text-gray-100 mb-4">
                "I watched too many small businesses get trapped in unfavorable contracts simply 
                because they couldn't afford quality legal advice. Technology should level the 
                playing field."
              </p>
              <div className="text-sm opacity-80">
                - Rajesh Kumar, Founder & CEO
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Values Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-6">
              Our Values
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              These principles guide everything we do, from product development to customer support.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {values.map((value, index) => (
              <motion.div
                key={value.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white p-8 rounded-xl shadow-lg text-center"
              >
                <value.icon className="w-12 h-12 text-primary-600 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-gray-900 mb-3">{value.title}</h3>
                <p className="text-gray-600">{value.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-6">
              Meet Our Team
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              A passionate team of technologists, legal experts, and entrepreneurs 
              committed to democratizing legal insights.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {team.map((member, index) => (
              <motion.div
                key={member.name}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-gray-50 rounded-xl p-8 text-center"
              >
                <div className="w-24 h-24 bg-gray-300 rounded-full mx-auto mb-6"></div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">{member.name}</h3>
                <p className="text-primary-600 font-medium mb-3">{member.role}</p>
                <p className="text-gray-600 text-sm">{member.bio}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section className="py-20 bg-gray-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl lg:text-4xl font-bold mb-6">
              Powered by Advanced AI
            </h2>
            <p className="text-lg text-gray-300 max-w-2xl mx-auto">
              Our technology combines cutting-edge AI with deep legal expertise to deliver 
              accurate, contextual, and actionable insights.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                title: 'Google Gemini AI',
                description: 'Latest generative AI technology for natural language understanding and analysis'
              },
              {
                title: 'Indian Legal Context',
                description: 'Trained specifically on Indian contract law and MSME requirements'
              },
              {
                title: 'Continuous Learning',
                description: 'Our AI improves with every analysis, becoming smarter and more accurate'
              }
            ].map((tech, index) => (
              <motion.div
                key={tech.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="w-16 h-16 bg-primary-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Award className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-bold mb-3">{tech.title}</h3>
                <p className="text-gray-300">{tech.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default AboutPage;