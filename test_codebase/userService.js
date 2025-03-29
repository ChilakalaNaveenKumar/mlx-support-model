
// User API service
import axios from 'axios';
import { getAuthToken } from './authUtils';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://api.example.com';

/**
 * Fetch user data from the API
 * @param {string} userId - The user ID
 * @returns {Promise<Object>} - User data
 */
export const fetchUserData = async (userId) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/users/${userId}`, {
      headers: {
        'Authorization': `Bearer ${getAuthToken()}`
      }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching user data:', error);
    throw new Error('Failed to fetch user data');
  }
};

/**
 * Update user profile information
 * @param {string} userId - The user ID
 * @param {Object} profileData - The profile data to update
 * @returns {Promise<Object>} - Result of the update operation
 */
export const updateUserProfile = async (userId, profileData) => {
  try {
    const response = await axios.put(`${API_BASE_URL}/users/${userId}`, profileData, {
      headers: {
        'Authorization': `Bearer ${getAuthToken()}`,
        'Content-Type': 'application/json'
      }
    });
    
    return {
      success: response.status === 200,
      data: response.data
    };
  } catch (error) {
    console.error('Error updating profile:', error);
    throw new Error('Failed to update profile');
  }
};

/**
 * Fetch user activity feed
 * @param {string} userId - The user ID
 * @param {number} page - Page number for pagination
 * @param {number} limit - Number of items per page
 * @returns {Promise<Object>} - Activity feed data
 */
export const fetchUserActivity = async (userId, page = 1, limit = 10) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/users/${userId}/activity`, {
      params: { page, limit },
      headers: {
        'Authorization': `Bearer ${getAuthToken()}`
      }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching activity feed:', error);
    throw new Error('Failed to fetch activity feed');
  }
};

// This service depends on the authentication service
// to provide valid tokens for API requests
